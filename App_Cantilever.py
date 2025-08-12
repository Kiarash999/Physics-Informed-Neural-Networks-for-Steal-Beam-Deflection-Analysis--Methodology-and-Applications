import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Helper functions
def analytical_y(x, L, E, I, P):
    # Analytical solution for beam deflection
    return (P / (6 * E * I)) * (3 * L * x**2 - x**3)

def fem_solution(x, L, E, I, P):
    # FEM solution for beam deflection
    return (P * L**3) / (3 * E * I) * (x / L)**2 * (3 - x / L) / 2

def create_constrained_model():
    # Build a PINN model with a hard constraint enforcing y(0)=0.
    # The final output is multiplied by the input so that at x=0, y_pred=0 regardless of NN output.
    inputs = tf.keras.Input(shape=(1,))
    x = tf.keras.layers.Dense(40, activation='tanh')(inputs)
    x = tf.keras.layers.Dense(40, activation='tanh')(x)
    x = tf.keras.layers.Dense(40, activation='tanh')(x)
    x = tf.keras.layers.Dense(1)(x)
    # Hard constraint: y_pred = x * NN(x), ensuring y(0)=0 always.
    outputs = tf.multiply(inputs, x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Main PINN function
def run_pinn(L, E, I, P, num_data_points, num_colloc_points, num_epochs, optimizer_choice):
    try:
        # Generate training data with noise
        x_data = np.linspace(0, L, num_data_points).reshape(-1, 1)
        y_data = analytical_y(x_data, L, E, I, P) + 0.1 * np.random.randn(*x_data.shape)
        # Collocation points for physics loss
        x_colloc = np.linspace(0, L, num_colloc_points).reshape(-1, 1)

        # Build PINN model with hard constraint at x=0
        model = create_constrained_model()

        # Convert data to tensors
        x_train = tf.convert_to_tensor(x_data, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_data, dtype=tf.float32)
        x_colloc_tensor = tf.convert_to_tensor(x_colloc, dtype=tf.float32)

        # Choose optimizer with learning rate 0.001
        if optimizer_choice == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        elif optimizer_choice == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        elif optimizer_choice == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

        # Initialize loss history dictionary
        loss_history = {'total': [], 'data': [], 'physics': []}

        # Clear and update progress text
        progress_text.delete("1.0", tk.END)
        progress_text.insert(tk.END, "Starting training...\n")
        progress_text.insert(tk.END, "Please do not click during training.\n")
        progress_bar['value'] = 0
        window.update_idletasks()

        # Training loop
        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                # Data loss term
                y_pred = model(x_train)
                loss_data = tf.reduce_mean(tf.square(y_pred - y_train))
                
                # Physics loss term: enforce the beam equilibrium equation
                with tf.GradientTape(persistent=True) as tape2:
                    tape2.watch(x_colloc_tensor)
                    with tf.GradientTape() as tape1:
                        tape1.watch(x_colloc_tensor)
                        y_phys = model(x_colloc_tensor)
                    dy_dx = tape1.gradient(y_phys, x_colloc_tensor)
                d2y_dx2 = tape2.gradient(dy_dx, x_colloc_tensor)
                residual = E * I * d2y_dx2 - P * (L - x_colloc_tensor)
                loss_physics = tf.reduce_mean(tf.square(residual))
                
                # Total loss combining data and physics losses
                total_loss = loss_data + 0.5 * loss_physics
                
                loss_history['total'].append(total_loss.numpy())
                loss_history['data'].append(loss_data.numpy())
                loss_history['physics'].append(loss_physics.numpy())

            # Compute gradients and update weights
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Update progress bar
            progress_bar['value'] = (epoch + 1) / num_epochs * 100
            window.update_idletasks()

            # Display progress every 50 epochs
            if (epoch + 1) % 50 == 0 or (epoch + 1) == num_epochs:
                progress_text.insert(tk.END, 
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Total: {total_loss.numpy():.2e}, "
                    f"Data: {loss_data.numpy():.2e}, "
                    f"Physics: {loss_physics.numpy():.2e}\n")
                progress_text.see(tk.END)

        # Increase resolution for smoother prediction curve
        x_test = np.linspace(0, L, 500).reshape(-1, 1)
        x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
        y_pred = model(x_test_tensor).numpy()
        y_analytical = analytical_y(x_test, L, E, I, P)
        y_fem = fem_solution(x_test, L, E, I, P)

        # Compute derivatives for physics validation
        with tf.GradientTape() as tape2:
            tape2.watch(x_test_tensor)
            with tf.GradientTape() as tape1:
                tape1.watch(x_test_tensor)
                y_phys = model(x_test_tensor)
            dy_dx = tape1.gradient(y_phys, x_test_tensor)
        d2y_dx2 = tape2.gradient(dy_dx, x_test_tensor)
        d2y_analytical = (P / (E * I)) * (L - x_test)

        # Scale deflection from meters to millimeters
        scale_to_mm = 1000.0
        
        # Plotting the results in multiple subplots
        plt.figure(figsize=(15, 15))
        
        # 1. Beam Deflection (with training and collocation points) - scaled in mm
        plt.subplot(3, 2, 1)
        plt.plot(x_test, y_analytical * scale_to_mm, 'b-', label='Analytical (mm)')
        plt.plot(x_test, y_pred * scale_to_mm, 'r--', label='PINN (mm)')
        plt.plot(x_test, y_fem * scale_to_mm, 'm:', label='FEM (mm)')
        plt.scatter(x_data, y_data * scale_to_mm, c='k', label='Training Data (mm)')
        plt.scatter(x_colloc, np.zeros_like(x_colloc), c='green', marker='x', label='Collocation (mm)')
        plt.xlabel('Position x (m)')
        plt.ylabel('Deflection y (mm)')
        plt.title('Beam Deflection (Scaled to mm)')
        plt.legend()
        plt.grid(True)
        
        # 2. Prediction Error (scaled in mm)
        plt.subplot(3, 2, 2)
        plt.plot(x_test, (y_pred - y_analytical) * scale_to_mm, 'g-', label='Error (mm)')
        plt.xlabel('Position x (m)')
        plt.ylabel('Error (mm)')
        plt.title('Prediction Error (Scaled to mm)')
        plt.legend()
        plt.grid(True)
        
        # 3. Second Derivative Comparison (no scaling)
        plt.subplot(3, 2, 3)
        plt.plot(x_test, d2y_analytical, 'b-', label='Analytical')
        plt.plot(x_test, d2y_dx2.numpy(), 'r--', label='PINN')
        plt.xlabel('Position x (m)')
        plt.ylabel('Second Derivative d2y/dx2')
        plt.title('Second Derivative Comparison')
        plt.legend()
        plt.grid(True)
        
        # 4. Training Loss History
        plt.subplot(3, 2, 4)
        plt.semilogy(loss_history['total'], label='Total Loss')
        plt.semilogy(loss_history['data'], label='Data Loss')
        plt.semilogy(loss_history['physics'], label='Physics Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.legend()
        plt.grid(True)
        
        # 5. Deflection Curve (scaled to mm)
        plt.subplot(3, 2, 5)
        plt.plot(x_test, y_analytical * scale_to_mm, 'b-', label='Analytical (mm)')
        plt.plot(x_test, y_pred * scale_to_mm, 'r--', label='PINN (mm)')
        plt.plot(x_test, y_fem * scale_to_mm, 'm:', label='FEM (mm)')
        plt.xlabel('Position x (m)')
        plt.ylabel('Deflection y (mm)')
        plt.title('Deflection Curve (Scaled to mm)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Function to start PINN training when button is clicked
def run_button_clicked():
    try:
        L = float(entry_L.get())
        E = float(entry_E.get())
        I = float(entry_I.get())
        P = float(entry_P.get())
        num_data_points = int(entry_data_points.get())
        num_colloc_points = int(entry_colloc_points.get())
        num_epochs = int(entry_epochs.get())
        optimizer_choice = optimizer_var.get()
        run_pinn(L, E, I, P, num_data_points, num_colloc_points, num_epochs, optimizer_choice)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers!")

# Main window configuration
window = tk.Tk()
window.title("Beam Deflection Prediction")
window.geometry("700x600")
window.configure(bg="#2C3E50")  # Dark theme

# Fonts configuration
font_label = ("Arial", 10, "bold")
font_entry = ("Arial", 10)
font_button = ("Arial", 12, "bold")

# Greeting label
tk.Label(window, text="Hi Professor Mirzabozorg, this is for you. K :)", font=("Arial", 14, "italic"), bg="#2C3E50", fg="white").pack(pady=5)

# Input frame for parameters
input_frame = tk.Frame(window, bg="#2C3E50")
input_frame.pack(pady=5)

tk.Label(input_frame, text="Length (L):", font=font_label, bg="#2C3E50", fg="white").grid(row=0, column=0, padx=5, pady=3, sticky="e")
entry_L = ttk.Entry(input_frame, font=font_entry, width=15)
entry_L.grid(row=0, column=1, padx=5, pady=3)

tk.Label(input_frame, text="Young's Modulus (E):", font=font_label, bg="#2C3E50", fg="white").grid(row=1, column=0, padx=5, pady=3, sticky="e")
entry_E = ttk.Entry(input_frame, font=font_entry, width=15)
entry_E.grid(row=1, column=1, padx=5, pady=3)

tk.Label(input_frame, text="Moment of Inertia (I):", font=font_label, bg="#2C3E50", fg="white").grid(row=2, column=0, padx=5, pady=3, sticky="e")
entry_I = ttk.Entry(input_frame, font=font_entry, width=15)
entry_I.grid(row=2, column=1, padx=5, pady=3)

tk.Label(input_frame, text="End Load (P):", font=font_label, bg="#2C3E50", fg="white").grid(row=3, column=0, padx=5, pady=3, sticky="e")
entry_P = ttk.Entry(input_frame, font=font_entry, width=15)
entry_P.grid(row=3, column=1, padx=5, pady=3)

tk.Label(input_frame, text="Number of Data Points:", font=font_label, bg="#2C3E50", fg="white").grid(row=4, column=0, padx=5, pady=3, sticky="e")
entry_data_points = ttk.Entry(input_frame, font=font_entry, width=15)
entry_data_points.grid(row=4, column=1, padx=5, pady=3)

tk.Label(input_frame, text="Number of Collocation Points:", font=font_label, bg="#2C3E50", fg="white").grid(row=5, column=0, padx=5, pady=3, sticky="e")
entry_colloc_points = ttk.Entry(input_frame, font=font_entry, width=15)
entry_colloc_points.grid(row=5, column=1, padx=5, pady=3)

tk.Label(input_frame, text="Number of Epochs:", font=font_label, bg="#2C3E50", fg="white").grid(row=6, column=0, padx=5, pady=3, sticky="e")
entry_epochs = ttk.Entry(input_frame, font=font_entry, width=15)
entry_epochs.grid(row=6, column=1, padx=5, pady=3)

# Optimizer selection menu
tk.Label(input_frame, text="Optimizer:", font=font_label, bg="#2C3E50", fg="white").grid(row=7, column=0, padx=5, pady=3, sticky="e")
optimizer_var = tk.StringVar(value="Adam")
optimizer_menu = tk.OptionMenu(input_frame, optimizer_var, "Adam", "SGD", "RMSprop")
optimizer_menu.config(bg="#34495E", fg="white", activebackground="#1ABC9C", activeforeground="white")
optimizer_menu.grid(row=7, column=1, padx=5, pady=3)

# Run button to start training
tk.Button(window, text="Run and Plot", command=run_button_clicked, font=font_button, width=15, height=1, bg="#E74C3C", fg="white").pack(pady=5)

# Progress frame to display training logs
progress_frame = tk.Frame(window, bg="#2C3E50")
progress_frame.pack(pady=5)

tk.Label(progress_frame, text="Training Progress:", font=font_label, bg="#2C3E50", fg="white").pack(pady=(5, 0))
progress_text = tk.Text(progress_frame, width=90, height=20, font=("Arial", 10), bg="#34495E", fg="white")
progress_text.pack(side=tk.LEFT, padx=10, pady=(0, 5))

scrollbar = tk.Scrollbar(progress_frame, command=progress_text.yview, bg="#34495E")
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
progress_text['yscrollcommand'] = scrollbar.set

# Progress bar widget
progress_bar = ttk.Progressbar(window, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=5)

window.mainloop()
