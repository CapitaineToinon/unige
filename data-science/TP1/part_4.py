import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Define the line parameters and point A
    a, b, c = 3, 4, 6
    point = np.array([-1., 3.], dtype=float)

    def line(x: float) -> float:
        return (-a * x - c) / b

    # Normal vector to the line
    normal_vector = np.array([a, b])

    # Compute the numerator as dot product plus the constant term
    num = np.dot(normal_vector, point) + c

    # Compute the denominator as the norm of the normal vector
    den = np.linalg.norm(normal_vector)

    # Distance from point to line
    distance = abs(num) / den

    print(f"dot product of normal vector and point: {num - c}")
    print(f"constant term: {c}")
    print(f"nominator (dot product + constant): {num}")
    print(f"denominator (norm of normal vector): {den}")
    print(f"distance: {distance}")

    # Calculate the perpendicular point projection onto the line
    x_perp = (b * (b * point[0] - a * point[1]) - a * c) / (a ** 2 + b ** 2)
    y_perp = (a * (a * point[1] - b * point[0]) - b * c) / (a ** 2 + b ** 2)

    x_vals = np.linspace(-10, 10, 400)
    y_vals = line(x_vals)

    plt.plot(x_vals, y_vals, label='3x + 4y + 6 = 0')

    # Plot point A
    plt.scatter(point[0], point[1], color='red', zorder=5)
    plt.text(point[0], point[1], '(-1, 3)', fontsize=12, verticalalignment='bottom')

    # Plot the perpendicular line projection
    plt.scatter(x_perp, y_perp, color='blue', zorder=5)
    plt.plot([point[0], x_perp], [point[1], y_perp], linestyle='--', color='gray',
             label=f'Distance = {round(distance, 2)}')

    # Annotation for the projected point
    plt.text(x_perp, y_perp, 'P(Projection)', fontsize=12, verticalalignment='bottom')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.title('Projection of Point A onto Line')
    plt.show()
