
import numpy as np
import matplotlib.pyplot as plt

def calculate(img, num_samples=500):

    img = img > .5  # Ensure (very naively) the image is binary

    edge_y, edge_x = np.where(img)
    points = np.column_stack((edge_x, edge_y))
    ordered_pts = [points[0]]
    unvisited = list(points[1:])
    while unvisited:
        last_pt = ordered_pts[-1]
        # Calculate Euclidean distances to remaining unvisited edge pixels
        dists = np.sum((unvisited - last_pt)**2, axis=1)
        nearest_idx = np.argmin(dists)
        ordered_pts.append(unvisited.pop(nearest_idx))
    ordered_pts = np.array(ordered_pts)

    distances = np.sqrt(np.sum(np.diff(ordered_pts, axis=0)**2, axis=1))
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_dist[-1]
    target_dists = np.linspace(0, total_length, num_samples, endpoint=False)

    x_resampled = np.interp(target_dists, cumulative_dist, ordered_pts[:, 0]).astype(int)
    y_resampled = np.interp(target_dists, cumulative_dist, ordered_pts[:, 1]).astype(int)

    return x_resampled, y_resampled

def plot_trajectory(x, y):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    total_points = len(x)

    for i, ax in enumerate(axes):
        percentage = (i + 1) * 10
        num_points = max(1, int(total_points * percentage / 100))

        ax.plot(
            x[:num_points],
            y[:num_points],
            color="blue",
            linewidth=1.5
        )

        ax.set_title(f"{percentage}% Complete")
        ax.set_xlim(min(x) - 10, max(x) + 10)
        ax.set_ylim(max(y) + 10, min(y) - 10)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    return fig