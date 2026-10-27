import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

def train_model(learning_rate):
    w = 0.6
    b = 0.0

    for iteration in range(1, 1201):
        predictions = w * x + b

        errors = predictions - y

        gradient_b = np.mean(errors)
        gradient_w = np.mean(x * errors)
        half_MSE = (np.mean(errors**2))/2
        if half_MSE < 0.000001:
            return iteration

        b = b - (learning_rate * gradient_b)
        w = w - (learning_rate * gradient_w)


learning_rates = np.arange(0.163, 0.168, 0.001)
best_lr = None
best_iterations = float("inf")


for lr in learning_rates:
    iterations = train_model(lr)
    print(lr, iterations)
    if iterations is not None and iterations < best_iterations:
        best_iterations = iterations
        best_lr = lr


print(f"Best lr, iterations : {best_lr, best_iterations}")