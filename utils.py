import numpy as np
import tensorflow as tf
import deepxde as dde
import matplotlib.pyplot as plt

def cantilever_uniform(x, L, E, I, q):
    return (q / (24 * E * I)) * (x**4 - 4 * L * x**3 + 6 * L**2 * x**2)

def fully_restrained_uniform(x, L, E, I, q):
    return (q / (24 * E * I)) * (x**4 - 2 * L * x**3 + L**3 * x)

def fully_restrained_point(x, L, E, I, P):
    w = np.zeros_like(x)
    mask = (x <= L/2)
    w[mask] = (P / (48 * E * I)) * (3 * L * x[mask]**2 - 4 * x[mask]**3)
    w[~mask] = (P / (48 * E * I)) * (3 * L * x[~mask]**2 - 4 * x[~mask]**3 + L**3 - 6 * L**2 * (x[~mask] - L/2))
    return w

def analytical_solution(x, L, E, I, P):
    return (P / (6 * E * I)) * (3 * L * x**2 - x**3)

def fem_solution(x, L, E, I, P):
    return (P * L**3) / (3 * E * I) * (x / L)**2 * (3 - x / L) / 2

def create_pinn_model(beam_type, L):
    inputs = tf.keras.Input(shape=(1,))
    x = tf.keras.layers.Dense(40, activation='tanh')(inputs)
    x = tf.keras.layers.Dense(40, activation='tanh')(x)
    x = tf.keras.layers.Dense(40, activation='tanh')(x)
    x = tf.keras.layers.Dense(1)(x)
    
    if beam_type=='cantilever':
        outputs = tf.multiply(inputs, x)
    elif beam_type=='fully_restrained':
        outputs = tf.multiply(inputs*(L - inputs), x)
    else:
        outputs = tf.multiply(inputs, x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def train_pinn_model(L, E, I, P, num_data_points, num_colloc_points, num_epochs,
                     optimizer_choice="Adam", lr=0.01, patience=50, min_delta=1e-4,
                     use_early_stopping=True, use_lr_finder=False, load_type='point', beam_type='cantilever', q=None):
    
    x_data = np.linspace(0, L, num_data_points).reshape(-1, 1)
    
    if beam_type=='cantilever':
        if load_type=='point':
            y_data = analytical_solution(x_data, L, E, I, P) + 0.1 * np.random.randn(*x_data.shape)
        elif load_type=='uniform':
            y_data = cantilever_uniform(x_data, L, E, I, q) + 0.1 * np.random.randn(*x_data.shape)
    elif beam_type=='fully_restrained':
        if load_type=='point':
            y_data = fully_restrained_point(x_data, L, E, I, P) + 0.1 * np.random.randn(*x_data.shape)
        elif load_type=='uniform':
            y_data = fully_restrained_uniform(x_data, L, E, I, q) + 0.1 * np.random.randn(*x_data.shape)
    
    x_colloc = np.linspace(0, L, num_colloc_points).reshape(-1, 1)
    model = create_pinn_model(beam_type, L)
    
    x_train = tf.convert_to_tensor(x_data, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_data, dtype=tf.float32)
    x_colloc_tensor = tf.convert_to_tensor(x_colloc, dtype=tf.float32)
    
    if optimizer_choice == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_choice == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer_choice == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        raise ValueError("Unsupported optimizer.")
    
    if use_lr_finder:
        min_lr = 1e-6
        max_lr = 1e-1
        num_iterations = 100
        factor = (max_lr / min_lr) ** (1 / (num_iterations - 1))
        current_lr = min_lr
        losses = []
        lrs = []
        
        for iter in range(num_iterations):
            optimizer.learning_rate.assign(current_lr)
            with tf.GradientTape() as tape:
                y_pred = model(x_train)
                loss_data = tf.reduce_mean(tf.square(y_pred - y_train))
                
                with tf.GradientTape(persistent=True) as tape2:
                    tape2.watch(x_colloc_tensor)
                    with tf.GradientTape() as tape1:
                        tape1.watch(x_colloc_tensor)
                        y_phys = model(x_colloc_tensor)
                    dy_dx = tape1.gradient(y_phys, x_colloc_tensor)
                d2y_dx2 = tape2.gradient(dy_dx, x_colloc_tensor)
                
                if beam_type=='cantilever':
                    if load_type=='point':
                        residual = E * I * d2y_dx2 - P * (L - x_colloc_tensor)
                    elif load_type=='uniform':
                        residual = E * I * d2y_dx2 - (q * (L - x_colloc_tensor)**2) / 2
                elif beam_type=='fully_restrained':
                    if load_type=='point':
                        f = tf.where(x_colloc_tensor<=L/2, (P/8.0)*(L - 4*x_colloc_tensor), (P/48.0)*(6*L - 24*x_colloc_tensor))
                        residual = E * I * d2y_dx2 - f
                    elif load_type=='uniform':
                        residual = E * I * d2y_dx2 - (q/2.0)*x_colloc_tensor*(x_colloc_tensor - L)
                
                loss_physics = tf.reduce_mean(tf.square(residual))
                total_loss = loss_data + 0.5 * loss_physics
            
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            losses.append(total_loss.numpy())
            lrs.append(current_lr)
            current_lr *= factor
        
        index_min = np.argmin(losses)
        optimal_lr = lrs[index_min]
        optimizer.learning_rate.assign(optimal_lr)
        print(f"Optimal learning rate found: {optimal_lr}")
    
    loss_history = {"total": [], "data": [], "physics": []}
    best_loss = np.inf
    wait = 0
    
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            y_pred = model(x_train)
            loss_data = tf.reduce_mean(tf.square(y_pred - y_train))
            
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x_colloc_tensor)
                with tf.GradientTape() as tape1:
                    tape1.watch(x_colloc_tensor)
                    y_phys = model(x_colloc_tensor)
                dy_dx = tape1.gradient(y_phys, x_colloc_tensor)
            d2y_dx2 = tape2.gradient(dy_dx, x_colloc_tensor)
            
            if beam_type=='cantilever':
                if load_type=='point':
                    residual = E * I * d2y_dx2 - P * (L - x_colloc_tensor)
                elif load_type=='uniform':
                    residual = E * I * d2y_dx2 - (q * (L - x_colloc_tensor)**2) / 2
            elif beam_type=='fully_restrained':
                if load_type=='point':
                    f = tf.where(x_colloc_tensor<=L/2, (P/8.0)*(L - 4*x_colloc_tensor), (P/48.0)*(6*L - 24*x_colloc_tensor))
                    residual = E * I * d2y_dx2 - f
                elif load_type=='uniform':
                    residual = E * I * d2y_dx2 - (q/2.0)*x_colloc_tensor*(x_colloc_tensor - L)
            
            loss_physics = tf.reduce_mean(tf.square(residual))
            total_loss = loss_data + 0.5 * loss_physics
        
        loss_history["total"].append(total_loss.numpy())
        loss_history["data"].append(loss_data.numpy())
        loss_history["physics"].append(loss_physics.numpy())
        
        if use_early_stopping:
            if total_loss < best_loss - min_delta:
                best_loss = total_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if (epoch + 1) % 50 == 0 or (epoch + 1) == num_epochs:
            print(f"Epoch {epoch+1}/{num_epochs}: Total Loss = {total_loss.numpy():.2e}, Data Loss = {loss_data.numpy():.2e}, Physics Loss = {loss_physics.numpy():.2e}")
    
    x_test = np.linspace(0, L, 500).reshape(-1, 1)
    x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test_pred = model(x_test_tensor).numpy()
    
    if beam_type=='cantilever':
        if load_type=='point':
            y_test_analytical = analytical_solution(x_test, L, E, I, P)
            d2y_analytical = (P / (E * I)) * (L - x_test)
        elif load_type=='uniform':
            y_test_analytical = cantilever_uniform(x_test, L, E, I, q)
            d2y_analytical = (q / (2 * E * I)) * (L - x_test)**2
        y_test_fem = fem_solution(x_test, L, E, I, P) if load_type=='point' else np.zeros_like(x_test)
    elif beam_type=='fully_restrained':
        if load_type=='point':
            y_test_analytical = fully_restrained_point(x_test, L, E, I, P)
            d2y_analytical = np.where(x_test<=L/2, (P/8.0)*(L - 4*x_test), (P/48.0)*(6*L - 24*x_test))
        elif load_type=='uniform':
            y_test_analytical = fully_restrained_uniform(x_test, L, E, I, q)
            d2y_analytical = (q / (2 * E * I)) * x_test * (x_test - L)
        y_test_fem = np.zeros_like(x_test)
    
    with tf.GradientTape() as tape2:
        tape2.watch(x_test_tensor)
        with tf.GradientTape() as tape1:
            tape1.watch(x_test_tensor)
            y_phys_test = model(x_test_tensor)
        dy_dx_test = tape1.gradient(y_phys_test, x_test_tensor)
    d2y_dx2_test = tape2.gradient(dy_dx_test, x_test_tensor)
    
    results = {
        "model": model,
        "loss_history": loss_history,
        "x_data": x_data,
        "y_data": y_data,
        "x_colloc": x_colloc,
        "x_test": x_test,
        "y_test_pred": y_test_pred,
        "y_test_analytical": y_test_analytical,
        "y_test_fem": y_test_fem,
        "d2y_dx2": d2y_dx2_test,
        "d2y_analytical": d2y_analytical,
    }
    
    return results

def plot_results(results, scale_to_mm=1000.0):
    x_test = results["x_test"]
    y_pred = results["y_test_pred"]
    y_analytical = results["y_test_analytical"]
    x_data = results["x_data"]
    y_data = results["y_data"]
    loss_history = results["loss_history"]
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    axs[0, 0].plot(x_test, y_analytical * scale_to_mm, 'b-', label="Analytical")
    axs[0, 0].plot(x_test, y_pred * scale_to_mm, 'r--', label="PINN Prediction")
    axs[0, 0].scatter(x_data, y_data * scale_to_mm, c='k', label="Training Data")
    axs[0, 0].set_xlabel("Position (m)")
    axs[0, 0].set_ylabel("Deflection (mm)")
    axs[0, 0].set_title("Deflection Profile Comparison")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    max_idx = np.argmax(np.abs(y_analytical))
    axs[0, 0].plot(x_test[max_idx], y_analytical[max_idx]*scale_to_mm, 'ko')
    axs[0, 0].annotate("Max Deflection",
                       xy=(x_test[max_idx], y_analytical[max_idx]*scale_to_mm),
                       xytext=(x_test[max_idx]+0.1, y_analytical[max_idx]*scale_to_mm),
                       arrowprops=dict(facecolor='black', shrink=0.05))
    
    axs[0, 1].semilogy(loss_history["total"], label="Total Loss")
    axs[0, 1].semilogy(loss_history["data"], label="Data Loss")
    axs[0, 1].semilogy(loss_history["physics"], label="Physics Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].set_title("Training Loss History")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(x_test, results["d2y_analytical"], 'b-', label="Analytical d2y/dx2")
    axs[1, 0].plot(x_test, results["d2y_dx2"], 'r--', label="PINN d2y/dx2")
    axs[1, 0].set_xlabel("Position (m)")
    axs[1, 0].set_ylabel("d2y/dx2")
    axs[1, 0].set_title("Second Derivative Comparison")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    error = (y_pred - y_analytical) * scale_to_mm
    axs[1, 1].plot(x_test, error, 'g-', label="Error")
    axs[1, 1].set_xlabel("Position (m)")
    axs[1, 1].set_ylabel("Error (mm)")
    axs[1, 1].set_title("Prediction Error")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    rmse = np.sqrt(np.mean((y_pred - y_analytical)**2))
    print(f"RMSE between PINN and Analytical: {rmse*scale_to_mm:.2f} mm")
