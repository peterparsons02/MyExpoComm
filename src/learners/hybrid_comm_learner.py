import copy
from components.episode_buffer import EpisodeBatch
import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from learners.q_learner import QLearner

class HybridCommLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)

        self.global_reward = getattr(args, "global_reward", False) or (args.env == "struct_marl")
        
        # --- Configs ---
        self.aux_coef = args.aux_coef
        self.cont_t_interval = args.topk_neighbors - 1
        self.temperature = getattr(args, "temperature", 0.1)
        self.neg_num = getattr(args, "neg_num", 1)
        
        self.mixer = None 
        self.target_mixer = None

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # Get the relevant quantities
        states = batch["state"][:, :-1]
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # --- FIX: Ensure Shapes for Broadcasting ---
        # Reshape [Batch, Time] -> [Batch, Time, 1] to match [Batch, Time, Agents]
        if rewards.dim() == 2: 
            rewards = rewards.unsqueeze(-1)
        if terminated.dim() == 2:
            terminated = terminated.unsqueeze(-1)

        # Standardise rewards if needed
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # --- 1. Forward Pass ---
        mac_out = []
        state_predicts = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, state_predicted = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            state_predicts.append(state_predicted)
        
        mac_out = th.stack(mac_out, dim=1)           
        state_predicts = th.stack(state_predicts, dim=1) 

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # --- 2. Target Pass ---
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        target_mac_out = th.stack(target_mac_out[1:], dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Double Q-Learning
        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # ==============================================================================
        # >>> EXPANDED FIX BLOCK (Replace the previous block with this) <<<
        # ==============================================================================
        
        # 1. Fix target_max_qvals: [Batch, Time, Agents] -> [Batch, Time, Agents, 1]
        if target_max_qvals.dim() == 3:
            target_max_qvals = target_max_qvals.unsqueeze(-1)

        # 2. Fix chosen_action_qvals: [Batch, Time, Agents] -> [Batch, Time, Agents, 1]
        if chosen_action_qvals.dim() == 3:
            chosen_action_qvals = chosen_action_qvals.unsqueeze(-1)
            
        # 3. Fix terminated: [Batch, Time, 1] -> [Batch, Time, 1, 1]
        if terminated.dim() == 3:
            terminated = terminated.unsqueeze(-1)
        elif terminated.dim() == 2:
            terminated = terminated.unsqueeze(-1).unsqueeze(-1)

        # 4. Fix mask: [Batch, Time, 1] -> [Batch, Time, 1, 1]
        # >>> THIS IS THE NEW PART <<<
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(-1).unsqueeze(-1)

        # --- 3. Calculate Independent TD Error ---
        if self.global_reward:
            # THIS IS WHERE IT WAS CRASHING
            # rewards [B,T,1] + [B,T,Agents] works. 
            # rewards [B,T] + [B,T,Agents] crashes.
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()
            td_error = chosen_action_qvals - targets.detach()
        else:
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()[:, :, :, None]
            td_error = chosen_action_qvals[:, :, :, None] - targets.detach()

        mask_expand = mask.expand_as(td_error)
        masked_td_error = td_error * mask_expand
        
        # Main Q-Loss
        q_loss = (masked_td_error ** 2).sum() / mask_expand.sum()

        # --- 4. Contrastive Loss (InfoNCE) ---
        info_nce_loss = self.info_nce_loss(state_predicts[:, :-1], mask.squeeze(-1)) # FIX: added .squeeze(-1)

        # Adaptive Loss Scaling
        eps = 1e-10
        loss = (
            q_loss
            + self.aux_coef
            * (q_loss / (info_nce_loss + eps)).abs().detach()
            * info_nce_loss
        )

        # --- 5. Optimize ---
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # Update Target Networks
        self.training_steps += 1
        if (self.args.target_update_interval_or_tau > 1 and 
            (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        # Logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("info_nce_loss", info_nce_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.log_stats_t = t_env

    # --- InfoNCE Loss ---
    def info_nce_loss(self, predict_feat, mask):
        bs, n_steps, n_agents, feat_dim = predict_feat.size()
        
        ref_timestep = th.randint(0, n_steps, (1,)).to(predict_feat.device)
        ref_feat = predict_feat[:, ref_timestep[0]]
        ref_mask = mask[:, ref_timestep[0]][:, None, :].expand(-1, n_agents, -1)

        pos_ind_temp = th.arange(n_agents).to(predict_feat.device)[None, :].expand(bs, -1)
        pos_ind = th.randint(0, n_agents - 1, (bs, n_agents)).to(predict_feat.device)
        pos_ind[pos_ind >= pos_ind_temp] += 1
        pos_ind = pos_ind[:, :, None].expand(-1, -1, feat_dim)
        
        pos_feat = predict_feat[:, ref_timestep[0]]
        pos_feat = th.gather(pos_feat, 1, pos_ind)

        neg_time_ind_candidates = []
        for x in range(n_steps):
            if x < ref_timestep - self.cont_t_interval or x > ref_timestep + self.cont_t_interval:
                neg_time_ind_candidates.append(x)
        
        if len(neg_time_ind_candidates) == 0:
            return th.tensor(0.0).to(predict_feat.device)

        neg_time_ind_candidates = th.tensor(neg_time_ind_candidates).to(predict_feat.device)
        ind_ind = th.randint(0, len(neg_time_ind_candidates), (bs,)).to(predict_feat.device)
        neg_time_ind = neg_time_ind_candidates[ind_ind]
        
        neg_mask = mask[th.arange(bs), neg_time_ind, 0]
        neg_mask = neg_mask[:, None].expand(-1, n_agents)
        neg_time_ind = neg_time_ind[:, None, None, None].expand(-1, -1, n_agents, feat_dim)
        
        neg_feat = th.gather(predict_feat, 1, neg_time_ind)[:, 0, :, :]

        neg_agent_ind = th.stack([th.randperm(n_agents)[: self.neg_num] for _ in range(bs)]).to(predict_feat.device)
        neg_agent_ind = neg_agent_ind[:, None, :, None].expand(-1, n_agents, -1, feat_dim)
        neg_feat = neg_feat[:, None, :, :].expand(-1, n_agents, -1, -1)
        neg_feat = th.gather(neg_feat, 2, neg_agent_ind)

        ref_feat = ref_feat.reshape(bs * n_agents, feat_dim)
        pos_feat = pos_feat.reshape(bs * n_agents, feat_dim)
        neg_feat = neg_feat.reshape(bs * n_agents, self.neg_num, feat_dim)
        ref_mask = ref_mask.reshape(bs * n_agents)
        neg_mask = neg_mask.reshape(bs * n_agents)

        pos_pairs = th.matmul(pos_feat[:, None, :], ref_feat[:, :, None]).squeeze(2)
        neg_pairs = th.matmul(pos_feat[:, None, :], neg_feat.permute(0, 2, 1)).squeeze(1)

        logits = th.cat([pos_pairs, neg_pairs], dim=1)
        final_mask = (ref_mask * neg_mask).bool()

        logits = logits[final_mask]
        labels = th.zeros(logits.shape[0], dtype=th.long).to(predict_feat.device)

        logits = logits / self.temperature
        return F.cross_entropy(logits, labels)