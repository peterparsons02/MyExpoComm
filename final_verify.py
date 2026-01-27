import torch
import gym
import sys
import numpy as np

def run_integration_test():
    print("=" * 50)
    print("🚀 STARTING FINAL SYSTEM INTEGRATION TEST")
    print("=" * 50)

    # 1. CHECK GPU
    print(f"Step 1: Checking Hardware...")
    if not torch.cuda.is_available():
        print("❌ CRITICAL FAILURE: CUDA is not available. PyTorch cannot see the GPU.")
        sys.exit(1)
    
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU Detected: {gpu_name}")
    print(f"✅ PyTorch Version: {torch.__version__}")
    print(f"✅ CUDA Version: {torch.version.cuda}")

    # 2. INITIALIZE ENVIRONMENT
    print(f"\nStep 2: Initializing Gym Environment...")
    try:
        # Using CartPole because it's lightweight
        env = gym.make('CartPole-v0')
        obs = env.reset()
        print(f"✅ Gym Environment ({env.spec.id}) created successfully.")
    except Exception as e:
        print(f"❌ GYM FAILURE: {e}")
        sys.exit(1)

    # 3. RUN SIMULATION LOOP (CPU -> GPU Integration)
    print(f"\nStep 3: Testing CPU-to-GPU Data Pipeline...")
    try:
        # Create a dummy neural network layer on the GPU
        model = torch.nn.Linear(4, 2).to(device)
        
        # Step the environment (CPU)
        obs, reward, done, info = env.step(env.action_space.sample())
        
        # Move data to GPU and process
        obs_tensor = torch.from_numpy(obs).float().to(device)
        output = model(obs_tensor)
        
        # Move result back to CPU
        result = output.cpu().detach().numpy()
        
        print(f"✅ Data moved to GPU ({gpu_name}), processed, and returned.")
        print(f"   Input: {obs}")
        print(f"   Output: {result}")
        
    except Exception as e:
        print(f"❌ COMPUTE FAILURE: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("🎉 SUCCESS: YOU ARE GOOD TO GO!")
    print("=" * 50)

if __name__ == "__main__":
    run_integration_test()
