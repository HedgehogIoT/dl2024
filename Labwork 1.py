def gradient_descent(f, f_, x_init, learning_rate=0.1, max_iter=100):
    x = x_init  
    iteration = 0

    print(f"Iteration | Time | x       | f(x)")

    while iteration < max_iter:
        fx = f(x)

        print(f"{iteration:9d} | {iteration:4d} | {x: .2f} | {fx: .2f}")

        x_new = x - learning_rate * f_(x)
        x = x_new
        iteration += 1

    return x

def f(x):
    return x ** 2

def f_(x):
    return 2 * x

def main():
    x_init = 10.0
    learning_rate = 0.1
    max_iter = 11

    minimum_x = gradient_descent(f, f_, x_init, learning_rate, max_iter)


if __name__ == "__main__":
    main()