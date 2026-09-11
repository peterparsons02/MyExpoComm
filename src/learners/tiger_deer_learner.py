"""Short-batch safeguard, registered only by the Tiger-Deer launcher."""

from .hybrid_comm_learner import HybridCommLearner
from .q_learner import ContQLearner


class _ShortBatchGuard:
    def info_nce_loss(self, predict_feat, mask):
        # Every reference needs a negative more than cont_t_interval steps away.
        # Short batches skip only the auxiliary loss; inherited TD training runs.
        if predict_feat.size(1) <= 2 * self.cont_t_interval + 1:
            return predict_feat.new_zeros(())
        return super().info_nce_loss(predict_feat, mask)


class TigerDeerHybridCommLearner(_ShortBatchGuard, HybridCommLearner):
    pass


class TigerDeerContQLearner(_ShortBatchGuard, ContQLearner):
    pass
