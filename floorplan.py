import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Rectangle


class SimpleFloorplanGenerator:
    """Generate clean 2D architectural floorplan from labeled point cloud."""

    def __init__(self, xyz, labels):
        """
        Initialize with labeled point cloud.

        Parameters:
        - xyz: 3D point cloud (N, 3)
        - labels: point labels (N,)
        """
        self.xyz = xyz
        self.labels = labels

        # Label mapping
        self.LABEL_MAP = {
            "ceiling": 0,
            "floor": 1,
            "wall": 2,
            "beam": 3,
            "column": 4,
            "window": 5,
            "door": 6,
        }

        # Extract 2D projections for each element type
        self.extract_elements()

    def extract_elements(self):
        """Extract 2D projections of each architectural element."""
        # Project to 2D (remove Z coordinate)
        self.points_2d = self.xyz[:, :2]

        # Extract each element type
        self.walls = self.points_2d[self.labels == self.LABEL_MAP["wall"]]
        self.columns = self.points_2d[self.labels == self.LABEL_MAP["column"]]
        self.windows = self.points_2d[self.labels == self.LABEL_MAP["window"]]
        self.doors = self.points_2d[self.labels == self.LABEL_MAP["door"]]
        self.beams = self.points_2d[self.labels == self.LABEL_MAP["beam"]]

        print(f"Extracted elements:")
        print(f"  Walls: {len(self.walls)} points")
        print(f"  Columns: {len(self.columns)} points")
        print(f"  Windows: {len(self.windows)} points")
        print(f"  Doors: {len(self.doors)} points")
        print(f"  Beams: {len(self.beams)} points")

    def get_room_boundary(self):
        """Get the outer boundary of the room from wall points."""
        if len(self.walls) == 0:
            return None

        # Get the bounding box of wall points
        min_coords = self.walls.min(axis=0)
        max_coords = self.walls.max(axis=0)

        # Add small padding
        padding = 0.1
        min_coords -= padding
        max_coords += padding

        # Create rectangle corners (assuming rectangular room)
        corners = [
            [min_coords[0], min_coords[1]],
            [max_coords[0], min_coords[1]],
            [max_coords[0], max_coords[1]],
            [min_coords[0], max_coords[1]],
            [min_coords[0], min_coords[1]],  # Close the polygon
        ]

        return np.array(corners)

    def find_wall_segments_simple(self):
        """Find wall segments using a simpler approach."""
        if len(self.walls) == 0:
            return []

        # Get bounds
        xmin, ymin = self.walls.min(axis=0)
        xmax, ymax = self.walls.max(axis=0)

        # Define wall thickness threshold
        thickness = 0.3  # 30cm

        wall_segments = []

        # Bottom wall
        bottom_wall_points = self.walls[np.abs(self.walls[:, 1] - ymin) < thickness]
        if len(bottom_wall_points) > 50:
            x_start = bottom_wall_points[:, 0].min()
            x_end = bottom_wall_points[:, 0].max()
            wall_segments.append(
                {"start": [x_start, ymin], "end": [x_end, ymin], "type": "horizontal"}
            )

        # Top wall
        top_wall_points = self.walls[np.abs(self.walls[:, 1] - ymax) < thickness]
        if len(top_wall_points) > 50:
            x_start = top_wall_points[:, 0].min()
            x_end = top_wall_points[:, 0].max()
            wall_segments.append(
                {"start": [x_start, ymax], "end": [x_end, ymax], "type": "horizontal"}
            )

        # Left wall
        left_wall_points = self.walls[np.abs(self.walls[:, 0] - xmin) < thickness]
        if len(left_wall_points) > 50:
            y_start = left_wall_points[:, 1].min()
            y_end = left_wall_points[:, 1].max()
            wall_segments.append(
                {"start": [xmin, y_start], "end": [xmin, y_end], "type": "vertical"}
            )

        # Right wall
        right_wall_points = self.walls[np.abs(self.walls[:, 0] - xmax) < thickness]
        if len(right_wall_points) > 50:
            y_start = right_wall_points[:, 1].min()
            y_end = right_wall_points[:, 1].max()
            wall_segments.append(
                {"start": [xmax, y_start], "end": [xmax, y_end], "type": "vertical"}
            )

        return wall_segments

    def get_openings(self):
        """Get bounding boxes for doors and windows."""
        openings = {"doors": [], "windows": []}

        # Process doors
        if len(self.doors) > 0:
            door_min = self.doors.min(axis=0)
            door_max = self.doors.max(axis=0)
            door_center = (door_min + door_max) / 2
            door_size = door_max - door_min

            openings["doors"].append(
                {
                    "min": door_min,
                    "max": door_max,
                    "center": door_center,
                    "size": door_size,
                }
            )

        # Process windows
        if len(self.windows) > 0:
            window_min = self.windows.min(axis=0)
            window_max = self.windows.max(axis=0)
            window_center = (window_min + window_max) / 2
            window_size = window_max - window_min

            openings["windows"].append(
                {
                    "min": window_min,
                    "max": window_max,
                    "center": window_center,
                    "size": window_size,
                }
            )

        return openings

    def get_beam_segments(self):
        """Get beam positions (overhead structural elements)."""
        if len(self.beams) == 0:
            return []

        # Beams typically run along walls
        # Group beams by their primary direction
        beam_segments = []

        # Get bounds of beam points
        xmin, ymin = self.beams.min(axis=0)
        xmax, ymax = self.beams.max(axis=0)

        # Check if beams run horizontally (along top of walls)
        # Beams are usually near the walls, so we check proximity to wall boundaries
        wall_xmin, wall_ymin = self.walls.min(axis=0)
        wall_xmax, wall_ymax = self.walls.max(axis=0)

        threshold = 0.5  # 50cm proximity to walls

        # Top beam
        top_beams = self.beams[np.abs(self.beams[:, 1] - wall_ymax) < threshold]
        if len(top_beams) > 10:
            beam_segments.append(
                {
                    "start": [top_beams[:, 0].min(), wall_ymax],
                    "end": [top_beams[:, 0].max(), wall_ymax],
                    "type": "horizontal",
                }
            )

        # Bottom beam
        bottom_beams = self.beams[np.abs(self.beams[:, 1] - wall_ymin) < threshold]
        if len(bottom_beams) > 10:
            beam_segments.append(
                {
                    "start": [bottom_beams[:, 0].min(), wall_ymin],
                    "end": [bottom_beams[:, 0].max(), wall_ymin],
                    "type": "horizontal",
                }
            )

        # Left beam
        left_beams = self.beams[np.abs(self.beams[:, 0] - wall_xmin) < threshold]
        if len(left_beams) > 10:
            beam_segments.append(
                {
                    "start": [wall_xmin, left_beams[:, 1].min()],
                    "end": [wall_xmin, left_beams[:, 1].max()],
                    "type": "vertical",
                }
            )

        # Right beam
        right_beams = self.beams[np.abs(self.beams[:, 0] - wall_xmax) < threshold]
        if len(right_beams) > 10:
            beam_segments.append(
                {
                    "start": [wall_xmax, right_beams[:, 1].min()],
                    "end": [wall_xmax, right_beams[:, 1].max()],
                    "type": "vertical",
                }
            )

        return beam_segments

    def draw_floorplan(self, figsize=(12, 12)):
        """Draw clean architectural floorplan."""
        fig, ax = plt.subplots(figsize=figsize)

        # Get room boundary
        boundary = self.get_room_boundary()

        # Get wall segments
        wall_segments = self.find_wall_segments_simple()

        # Get beam segments
        beam_segments = self.get_beam_segments()

        # Get openings
        openings = self.get_openings()

        # Draw walls as thick black lines
        for wall in wall_segments:
            start = np.array(wall["start"])
            end = np.array(wall["end"])

            # Draw thick line for wall
            ax.plot([start[0], end[0]], [start[1], end[1]], "k-", linewidth=8)

        # Draw beams as dashed lines (parallel to walls, slightly offset inward)
        for beam in beam_segments:
            start = np.array(beam["start"])
            end = np.array(beam["end"])

            # Offset beam slightly inward from wall
            offset = 0.15  # 15cm inward
            if beam["type"] == "horizontal":
                if start[1] > 35:  # Top beam
                    start[1] -= offset
                    end[1] -= offset
                else:  # Bottom beam
                    start[1] += offset
                    end[1] += offset
            else:  # Vertical
                if start[0] < -18:  # Left beam
                    start[0] += offset
                    end[0] += offset
                else:  # Right beam
                    start[0] -= offset
                    end[0] -= offset

            # Draw beam as dashed line
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                "k--",
                linewidth=3,
                alpha=0.6,
                label="Beam" if "Beam" not in [l.get_label() for l in ax.lines] else "",
            )

        # Draw room boundary (if walls didn't form complete rectangle)
        if boundary is not None:
            # Draw as thin line where walls might be missing
            ax.plot(boundary[:, 0], boundary[:, 1], "k--", linewidth=1, alpha=0.3)

        # Draw columns as gray rectangles
        if len(self.columns) > 0:
            col_min = self.columns.min(axis=0)
            col_max = self.columns.max(axis=0)
            col_size = col_max - col_min
            rect = Rectangle(
                col_min,
                col_size[0],
                col_size[1],
                facecolor="gray",
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

        # Draw doors
        for door in openings["doors"]:
            # White rectangle for door opening
            rect = Rectangle(
                door["min"],
                door["size"][0],
                door["size"][1],
                facecolor="white",
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(rect)

            # Door swing arc
            door_width = max(door["size"]) * 0.8
            arc = Arc(
                door["center"],
                door_width * 2,
                door_width * 2,
                angle=0,
                theta1=0,
                theta2=90,
                color="red",
                linewidth=1.5,
                linestyle="--",
            )
            ax.add_patch(arc)

            # Door center point
            ax.plot(door["center"][0], door["center"][1], "ro", markersize=5)

        # Draw windows
        for window in openings["windows"]:
            # Determine orientation based on size
            if window["size"][0] > window["size"][1]:  # Horizontal
                # Draw as double horizontal lines
                ax.plot(
                    [window["min"][0], window["max"][0]],
                    [window["center"][1], window["center"][1]],
                    "b-",
                    linewidth=6,
                    alpha=0.7,
                )
            else:  # Vertical
                # Draw as double vertical lines
                ax.plot(
                    [window["center"][0], window["center"][0]],
                    [window["min"][1], window["max"][1]],
                    "b-",
                    linewidth=6,
                    alpha=0.7,
                )

        # Set axis properties
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
        ax.set_xlabel("X (meters)", fontsize=11)
        ax.set_ylabel("Y (meters)", fontsize=11)
        ax.set_title("Architectural Floor Plan", fontsize=14, fontweight="bold")

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="k", linewidth=3, label="Walls"),
            Line2D(
                [0],
                [0],
                color="k",
                linewidth=2,
                linestyle="--",
                alpha=0.6,
                label="Beams",
            ),
            patches.Patch(facecolor="gray", edgecolor="black", label="Columns"),
            patches.Patch(facecolor="white", edgecolor="red", label="Doors"),
            Line2D([0], [0], color="b", linewidth=3, alpha=0.7, label="Windows"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        plt.show()

    def draw_with_points(self, figsize=(12, 12)):
        """Draw floorplan with point clouds for debugging."""
        fig, ax = plt.subplots(figsize=figsize)

        # Plot point clouds
        ax.scatter(
            self.walls[:, 0],
            self.walls[:, 1],
            s=0.5,
            alpha=0.3,
            c="gray",
            label="Walls",
        )
        if len(self.beams) > 0:
            ax.scatter(
                self.beams[:, 0],
                self.beams[:, 1],
                s=0.5,
                c="brown",
                alpha=0.5,
                label="Beams",
            )
        if len(self.doors) > 0:
            ax.scatter(
                self.doors[:, 0],
                self.doors[:, 1],
                s=1,
                c="red",
                alpha=0.5,
                label="Doors",
            )
        if len(self.windows) > 0:
            ax.scatter(
                self.windows[:, 0],
                self.windows[:, 1],
                s=1,
                c="blue",
                alpha=0.5,
                label="Windows",
            )
        if len(self.columns) > 0:
            ax.scatter(
                self.columns[:, 0],
                self.columns[:, 1],
                s=2,
                c="orange",
                alpha=0.5,
                label="Columns",
            )

        # Get and draw wall segments
        wall_segments = self.find_wall_segments_simple()
        for wall in wall_segments:
            start = np.array(wall["start"])
            end = np.array(wall["end"])
            ax.plot(
                [start[0], end[0]], [start[1], end[1]], "k-", linewidth=3, alpha=0.7
            )

        # Get and draw beam segments
        beam_segments = self.get_beam_segments()
        for beam in beam_segments:
            start = np.array(beam["start"])
            end = np.array(beam["end"])
            ax.plot(
                [start[0], end[0]], [start[1], end[1]], "k--", linewidth=2, alpha=0.5
            )

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title("Floor Plan with Point Clouds")
        ax.legend()

        plt.tight_layout()
        plt.show()
