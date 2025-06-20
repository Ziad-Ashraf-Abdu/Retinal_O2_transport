import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from All_layers_Forward_Model import model

# ===========================
# FUNCTIONS TO CALCULATE ERRORS
# ===========================

def relative_l2_error(pred, ref):
    return np.linalg.norm(pred - ref) / np.linalg.norm(ref)

def evaluate_layer(ref, pred, layer_name):
    mae = mean_absolute_error(ref, pred)
    rmse = np.sqrt(mean_squared_error(ref, pred))
    rel_l2 = relative_l2_error(pred, ref)
    print(f"{layer_name}:")
    print(f"   ➔ MAE      = {mae:.4f}")
    print(f"   ➔ RMSE     = {rmse:.4f}")
    print(f"   ➔ Rel L2   = {rel_l2:.4f}\n")
    return mae, rmse, rel_l2

# ===========================
# REFERENCE DATA
# ===========================

ref_IR = np.array([
    26.000, 25.714, 25.519, 25.428, 25.441, 25.558, 25.779, 26.104, 26.533, 27.066, 27.703
])

ref_OR = np.array([
    27.716, 28.457, 29.315, 30.303, 31.408, 32.630, 33.982, 35.477, 37.115, 38.909, 40.846
])

ref_FL = np.array([
    40.846, 42.939, 45.214, 47.671, 50.310, 53.157, 56.212, 59.501, 63.024, 66.794, 70.824
])

ref_CC = np.array([
    70.837, 75.140, 79.755, 84.695, 89.960, 95.589, 101.608, 108.030, 114.881, 122.200, 130.000
])

# ===========================
# MEASUREMENT POINTS
# ===========================

zL_real, zIR_real, zOR_real, zFL_real, zCC_real = 0.0, 100.0, 150.0, 200.0, 250.0

z_meas_IR = np.linspace(zL_real, zIR_real, len(ref_IR)).reshape(-1,1)
z_meas_OR = np.linspace(zIR_real, zOR_real, len(ref_OR)).reshape(-1,1)
z_meas_FL = np.linspace(zOR_real, zFL_real, len(ref_FL)).reshape(-1,1)
z_meas_CC = np.linspace(zFL_real, zCC_real, len(ref_CC)).reshape(-1,1)

# ===========================
# MODEL PREDICTIONS 
# ===========================

u_pred_IR = model.predict(z_meas_IR).flatten()
u_pred_OR = model.predict(z_meas_OR).flatten()
u_pred_FL = model.predict(z_meas_FL).flatten()
u_pred_CC = model.predict(z_meas_CC).flatten()

# ===========================
# EVALUATION
# ===========================

print("======= Evaluation of Model Predictions =======\n")
evaluate_layer(ref_IR, u_pred_IR, "Inner Retina (IR)")
evaluate_layer(ref_OR, u_pred_OR, "Outer Retina (OR)")
evaluate_layer(ref_FL, u_pred_FL, "Photoreceptor Layer (FL)")
evaluate_layer(ref_CC, u_pred_CC, "Choroidal Capillary (CC)")

# ===========================
# OPTIONAL: Visualization
# ===========================

plt.figure(figsize=(12, 8))
plt.plot(z_meas_IR, ref_IR, 'o-', label='Reference IR', color='blue')
plt.plot(z_meas_IR, u_pred_IR, 'x--', label='Prediction IR', color='cyan')

plt.plot(z_meas_OR, ref_OR, 'o-', label='Reference OR', color='red')
plt.plot(z_meas_OR, u_pred_OR, 'x--', label='Prediction OR', color='orange')

plt.plot(z_meas_FL, ref_FL, 'o-', label='Reference FL', color='green')
plt.plot(z_meas_FL, u_pred_FL, 'x--', label='Prediction FL', color='lime')

plt.plot(z_meas_CC, ref_CC, 'o-', label='Reference CC', color='purple')
plt.plot(z_meas_CC, u_pred_CC, 'x--', label='Prediction CC', color='magenta')

plt.xlabel('Depth z (μm)')
plt.ylabel('Oxygen Partial Pressure (mmHg)')
plt.title('PINN Model Predictions vs Reference Data')
plt.legend()
plt.grid(True)
plt.show()