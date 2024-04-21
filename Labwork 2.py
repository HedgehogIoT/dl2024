def loss_function(w0, w1, x, y):
    error_sum = 0
    for i in range(len(x)):
        error_sum += (w1 * x[i] + w0 - y[i]) ** 2
    return error_sum / (2 * len(x))

def dL_dw0(w0, w1, x, y):
    error_sum = 0
    for i in range(len(x)):
        error_sum += w1 * x[i] + w0 - y[i]
    return error_sum / len(x)

def dL_dw1(w0, w1, x, y):
    error_sum = 0
    for i in range(len(x)):
        error_sum += x[i] * (w1 * x[i] + w0 - y[i])
    return error_sum / len(x)
def main():
    x = [1, 2, 3, 4, 5]  
    y = [2, 4, 6, 8, 10]  
    w0 = 0  
    w1 = 1  
    r = 0.01  
    t = 0.0001  
    max_iters = 1000 

    for i in range(max_iters):
        gradient_w0 = dL_dw0(w0, w1, x, y)
        gradient_w1 = dL_dw1(w0, w1, x, y)

        w0 = w0 - r * gradient_w0
        w1 = w1 - r * gradient_w1

        loss = loss_function(w0, w1, x, y)

        if abs(loss) < t:
            print(f"Converged after {i+1} iterations.")
            break

    print("Final w0:", w0)
    print("Final w1:", w1)


if __name__ == "__main__":
    main()

