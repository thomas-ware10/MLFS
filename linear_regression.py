import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])
loss_history = []

w = 0.6
b = 0.0
learning_rate = 0.164

for iteration in range(1, 1201):
    predictions = w * x + b

    errors = predictions - y

    gradient_b = np.mean(errors)
    gradient_w = np.mean(x * errors)
    half_MSE = (np.mean(errors**2))/2
    loss_history.append(half_MSE)
    if half_MSE < 0.000001:
        break

    b = b - (learning_rate * gradient_b)
    w = w - (learning_rate * gradient_w)


print(f"\n\nIterations completed : {iteration}")    
print(f"\nWeight : {w}\nBias : {b}\n")

new_predictions = w * x + b
print(f"New Predictions : {new_predictions}")

new_errors = new_predictions - y
print(f"New errors : {new_errors}")

print(f"\n\nHalf-MSE : {(np.mean(new_errors**2))/2}")



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(x, y, color="red",
                  label="Training Data")
ax1.plot(x, new_predictions, color="blue",
                             label="Model Predictions")

ax1.set_title("Linear Regression from Scratch", family="Arial",
                                            fontsize="25",
                                            fontweight="bold")


ax1.set_xlabel("Input (x)", fontsize=17)
ax1.set_ylabel("Target Value (Y)", fontsize=17)

ax1.grid(True)
ax1.legend()


# ax2.set_ylim(0, 0.05)
ax2.plot(loss_history, color="red")

ax2.set_xlabel("Training Iteration", fontsize=17)
ax2.set_ylabel("Half-MSE Loss", fontsize=17)

ax2.set_title("Gradient Descent - Loss Over Time", family="Arial",
                                               fontsize=22,
                                               fontweight="bold")
ax2.grid(True)
# plt.show()