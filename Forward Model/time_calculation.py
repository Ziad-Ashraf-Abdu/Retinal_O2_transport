import time
import numpy as np
from All_layers_Forward_Model import model, ref_IR, ref_OR, ref_FL, ref_CC, mae_ir, mae_or, mae_fl, mae_cc

# ================================
# TIMING CALCULATION FOR PINN MODEL
# ================================

def calculate_inference_time(model, test_points, num_runs=100):
    """
    Calculate the inference time for the PINN model
    
    Parameters:
    - model: Trained PINN model
    - test_points: Test points for inference
    - num_runs: Number of runs for averaging (default: 100)
    
    Returns:
    - avg_time: A1verage inference time in seconds
    - total_time: Total time for all runs
    - times_list: List of individual run times
    """
    
    times_list = []
    
    print(f"Running inference timing test with {num_runs} runs...")
    print(f"Test points shape: {test_points.shape}")
    
    # Warm-up run (to avoid initial overhead)
    _ = model.predict(test_points)
    
    # Multiple runs for accurate timing
    for i in range(num_runs):
        start_time = time.time()
        predictions = model.predict(test_points)
        end_time = time.time()
        
        run_time = end_time - start_time
        times_list.append(run_time)
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{num_runs} runs...")
    
    avg_time = np.mean(times_list)
    total_time = np.sum(times_list)
    std_time = np.std(times_list)
    
    return avg_time, total_time, times_list, std_time


# ================================
# RESULTS SUMMARY FOR TABLE
# ================================
print("\n" + "="*60)
print("SUMMARY FOR TABLE COMPLETION")
print("="*60)

print(f"{'Scenario':<10} {'Avg Time (s)':<15} {'Time/Point (ms)':<18} {'Std Dev (s)':<12}")
print("-" * 60)

# Example test points (replace with your actual arrays)
z_meas_IR = np.linspace(0, 100, 11).reshape(-1, 1)
z_meas_OR = np.linspace(100, 150, 11).reshape(-1, 1)
z_meas_FL = np.linspace(150, 200, 11).reshape(-1, 1)
z_meas_CC = np.linspace(200, 250, 11).reshape(-1, 1)

avg_time_ir, _, _, std_time_ir = calculate_inference_time(model, z_meas_IR)
avg_time_or, _, _, std_time_or = calculate_inference_time(model, z_meas_OR)
avg_time_fl, _, _, std_time_fl = calculate_inference_time(model, z_meas_FL)
avg_time_cc, _, _, std_time_cc = calculate_inference_time(model, z_meas_CC)

timing_results = {
    "IR": {"avg_time": avg_time_ir, "std_time": std_time_ir, "num_points": len(z_meas_IR)},
    "OR": {"avg_time": avg_time_or, "std_time": std_time_or, "num_points": len(z_meas_OR)},
    "FL": {"avg_time": avg_time_fl, "std_time": std_time_fl, "num_points": len(z_meas_FL)},
    "CC": {"avg_time": avg_time_cc, "std_time": std_time_cc, "num_points": len(z_meas_CC)},
}

for scenario, results in timing_results.items():
    avg_time = results['avg_time']
    std_time = results['std_time']
    time_per_point = avg_time / results['num_points'] * 1000
    
    print(f"{scenario:<10} {avg_time:<15.6f} {time_per_point:<18.4f} {std_time:<12.6f}")



# ================================
# CALCULATE TOTAL TIME AND RELATIVE ERROR
# ================================

# Calculate Total Time (sum of all layer times)
total_time_sum = (timing_results['IR']['avg_time'] + 
                  timing_results['OR']['avg_time'] + 
                  timing_results['FL']['avg_time'] + 
                  timing_results['CC']['avg_time'])

print(f"\nTOTAL TIME CALCULATION:")
print(f"IR Time: {timing_results['IR']['avg_time']:.4f} s")
print(f"OR Time: {timing_results['OR']['avg_time']:.4f} s") 
print(f"FL Time: {timing_results['FL']['avg_time']:.4f} s")
print(f"CC Time: {timing_results['CC']['avg_time']:.4f} s")
print(f"Total Time (sum): {total_time_sum:.4f} s")

# ================================
# CALCULATE RELATIVE ERROR FROM YOUR MODEL EVALUATION
# ================================
# Use the error values from your model evaluation
# (These should be available from your model training code)

# From your model evaluation results:
mae_ir_val = mae_ir if 'mae_ir' in locals() else 0.0  # Replace with your actual values
mae_or_val = mae_or if 'mae_or' in locals() else 0.0
mae_fl_val = mae_fl if 'mae_fl' in locals() else 0.0  
mae_cc_val = mae_cc if 'mae_cc' in locals() else 0.0

# Calculate relative errors (as percentages)
# Assuming you have reference values for each layer
ref_mean_IR = np.mean(ref_IR) if 'ref_IR' in locals() else 26.0  # Default values
ref_mean_OR = np.mean(ref_OR) if 'ref_OR' in locals() else 32.0
ref_mean_FL = np.mean(ref_FL) if 'ref_FL' in locals() else 55.0
ref_mean_CC = np.mean(ref_CC) if 'ref_CC' in locals() else 100.0

rel_error_IR = (mae_ir_val / ref_mean_IR) * 100 if ref_mean_IR > 0 else 0.0
rel_error_OR = (mae_or_val / ref_mean_OR) * 100 if ref_mean_OR > 0 else 0.0
rel_error_FL = (mae_fl_val / ref_mean_FL) * 100 if ref_mean_FL > 0 else 0.0
rel_error_CC = (mae_cc_val / ref_mean_CC) * 100 if ref_mean_CC > 0 else 0.0

# Total relative error (average of all layers)
total_rel_error = (rel_error_IR + rel_error_OR + rel_error_FL + rel_error_CC) / 4

print(f"\nRELATIVE ERROR CALCULATION:")
print(f"IR Relative Error: {rel_error_IR:.2f}%")
print(f"OR Relative Error: {rel_error_OR:.2f}%")
print(f"FL Relative Error: {rel_error_FL:.2f}%")
print(f"CC Relative Error: {rel_error_CC:.2f}%")
print(f"Total Relative Error (average): {total_rel_error:.2f}%")