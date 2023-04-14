# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Visualization code for plotly.
The function use prefix conventions in the following way:
    - 'get_*' functions (e.g., 'get_camera_frustums')
        return data that can be ploted with plotly
    - 'vis_*' functions (e.g., 'vis_camera_rays')
        return 'go.Figure' objects which are the plots. Go Figure! :')
"""

from typing import Any, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly import express as ex
from torchtyping import TensorType

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import Frustums, RayBundle
from nerfstudio.utils.math import Gaussians


def color_str(color):
    """Plotly color string.

    Args:
        color: list [r, g, b] in [0, 1] range

    Returns:
        str: plotly-formatted color string
    """
    color = list((np.array(color) * 255.0).astype("int"))
    return f"""rgb({color[0]}, {color[1]}, {color[2]})"""


def get_line_segments_from_lines(
    lines: TensorType["num_rays", 2, 3],
    color: str = color_str((1, 0, 0)),
    marker_color: str = color_str((1, 0, 0)),
    colors: Optional[List[str]] = None,
    draw_marker: bool = True,
    draw_line: bool = True,
    marker_size: float = 4,
    line_width: float = 10,
) -> List[Any]:
    """Returns a list of Scatter3D objects for creating lines with plotly.
    # TODO(ethan): make this function more efficient instead of having a list of objects.

    Args:
        lines: Tensor of lines.
        color: Color of the lines. Defaults to red.
        marker_color: Color of the markers. Defaults to red.
        colors: List of colors for each line. Defaults to None.
        draw_marker: Whether to draw markers. Defaults to True.
        draw_line: Whether to draw lines. Defaults to True.
        marker_size: Size of the markers. Defaults to 4.
        line_width: Width of the lines. Defaults to 10.

    Returns:
        Scatter3D object on lines.
    """
    data = []
    for idx, line in enumerate(lines):
        thiscolor = color if draw_line else "rgba(0, 0, 0, 0)"
        if colors is not None:
            marker_color = colors[idx]
            thiscolor = colors[idx]
        data.append(
            go.Scatter3d(
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                showlegend=False,
                marker=dict(
                    size=marker_size,
                    color=marker_color,
                    colorscale="Viridis",
                )
                if draw_marker
                else dict(color="rgba(0, 0, 0, 0)"),
                line=dict(color=thiscolor, width=line_width),
            )
        )
    return data


def vis_dataset(camera_origins: TensorType["num_cameras", 3], ray_bundle: RayBundle) -> go.FigureWidget:  # type: ignore
    """Visualize a dataset with plotly using our cameras and generated rays.

    Args:
        camera_origins: Tensor of camera origins.
        ray_bundle: Ray bundle.

    Returns:
        plotly figure.
    """

    skip = 1
    size = 8
    assert len(ray_bundle) < 500, "Let's not break plotly by plotting too many rays!"

    data = []
    data += [
        go.Scatter3d(
            x=camera_origins[::skip, 0],
            y=camera_origins[::skip, 1],
            z=camera_origins[::skip, 2],
            mode="markers",
            name="camera origins",
            marker=dict(color="rgba(0, 0, 0, 1)", size=size),
        )
    ]

    length = 2.0
    lines = torch.stack(
        [ray_bundle.origins, ray_bundle.origins + ray_bundle.directions * length], dim=1
    )  # (num_rays, 2, 3)

    data += get_line_segments_from_lines(lines)

    layout = go.Layout(
        autosize=False,
        width=1000,
        height=1000,
        margin=go.layout.Margin(l=50, r=50, b=100, t=100, pad=4),  # type: ignore
        scene=go.layout.Scene(  # type: ignore
            aspectmode="data",
            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25)),
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def get_random_color(colormap: Optional[List[str]] = None, idx: Optional[int] = None) -> str:
    """Get a random color from a colormap.

    Args:
        colormap: List of colors. Defaults to Plotly colors.
        idx: Index of color to return. Defaults to None.

    Returns:
        random color string
    """
    if colormap is None:
        colormap = ex.colors.qualitative.Plotly
    if idx is None:
        return colormap[np.random.randint(0, len(colormap))]
    return colormap[idx % len(colormap)]


def get_sphere(
    radius: float, center: TensorType[3] = None, color: str = "black", opacity: float = 1.0, resolution: int = 32
) -> go.Mesh3d:  # type: ignore
    """Returns a sphere object for plotting with plotly.

    Args:
        radius: radius of sphere.
        center: center of sphere. Defaults to origin.
        color: color of sphere. Defaults to "black".
        opacity: opacity of sphere. Defaults to 1.0.
        resolution: resolution of sphere. Defaults to 32.

    Returns:
        sphere object.
    """
    phi = torch.linspace(0, 2 * torch.pi, resolution)
    theta = torch.linspace(-torch.pi / 2, torch.pi / 2, resolution)
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")

    x = torch.cos(theta) * torch.sin(phi)
    y = torch.cos(theta) * torch.cos(phi)
    z = torch.sin(theta)
    pts = torch.stack((x, y, z), dim=-1)

    pts *= radius
    if center is not None:
        pts += center

    return go.Mesh3d(
        {
            "x": pts[:, :, 0].flatten(),
            "y": pts[:, :, 1].flatten(),
            "z": pts[:, :, 2].flatten(),
            "alphahull": 0,
            "opacity": opacity,
            "color": color,
        }
    )


def get_cube(
    side_length: float,
    center: TensorType[3] = None,
    color: str = "black",
    opacity: float = 1.0,
) -> go.Mesh3d:  # type: ignore
    """Returns a cube object for plotting with plotly.

    Args:
        side_length: side_length of cube.
        center: center of cube.
        color: color of cube.
        opacity: opacity of cube.

    Returns:
        cube object.
    """

    x = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    y = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
    z = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])

    pts = np.stack((x, y, z), axis=0)

    pts *= side_length / 2.0
    if center is not None:
        pts += center

    return go.Mesh3d(
        {
            "x": pts[0],
            "y": pts[1],
            "z": pts[2],
            "alphahull": 0,
            "opacity": opacity,
            "color": color,
        }
    )


def get_gaussian_ellipsiod(
    mean: TensorType[3],
    cov: TensorType[3, 3],
    n_std: int = 2,
    color="lightblue",
    opacity: float = 0.5,
    resolution: int = 20,
    name: str = "ellipse",
) -> go.Mesh3d:  # type: ignore
    """Get a plotly ellipsoid for a Gaussian.

    Args:
        mean: mean of the Gaussian.
        cov: covariance of the Gaussian.
        n_std: Standard devation to visualize. Defaults to 2 (95% confidence).
        color: Color of the ellipsoid. Defaults to None.
        opacity: Opacity of the ellipsoid. Defaults to 0.5.
        resolution: Resolution of the ellipsoid. Defaults to 20.
        name: Name of the ellipsoid. Defaults to "ellipse".

    Returns:
        ellipsoid object.
    """

    phi = torch.linspace(0, 2 * torch.pi, resolution)
    theta = torch.linspace(-torch.pi / 2, torch.pi / 2, resolution)
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")

    x = torch.cos(theta) * torch.sin(phi)
    y = torch.cos(theta) * torch.cos(phi)
    z = torch.sin(theta)
    pts = torch.stack((x, y, z), dim=-1)

    eigenvals, eigenvecs = torch.linalg.eigh(cov)
    idx = torch.sum(cov, dim=0).argsort()
    idx = eigenvals[idx].argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    scaling = torch.sqrt(eigenvals) * n_std
    pts = pts * scaling

    pts = pts @ eigenvecs.t()

    pts += mean

    return go.Mesh3d(
        {
            "x": pts[:, :, 0].flatten(),
            "y": pts[:, :, 1].flatten(),
            "z": pts[:, :, 2].flatten(),
            "alphahull": 0,
            "opacity": opacity,
            "color": color,
            "name": name,
        }
    )


def get_gaussian_ellipsoids_list(
    gaussians: Gaussians, opacity: float = 0.5, color: str = "random", resolution: int = 20
) -> List[Union[go.Mesh3d, go.Scatter3d]]:  # type: ignore
    """Get a list of plotly meshes for frustums.

    Args:
        gaussians (Gaussians): Gaussians to visualize.
        opacity (float, optional): Opacity of the mesh. Defaults to 0.3.
        color (str, optional): Color of the mesh. Defaults to "random".
        resolution: Resolution of the mesh. Defaults to 20.

    Returns:
        List of plotly meshes
    """
    data = []

    vis_means = go.Scatter3d(
        x=gaussians.mean[:, 0],
        y=gaussians.mean[:, 1],
        z=gaussians.mean[:, 2],
        mode="markers",
        marker=dict(size=2, color="black"),
        name="Means",
    )
    data.append(vis_means)

    for i in range(gaussians.mean.shape[0]):
        if color == "random":
            c = get_random_color()
        else:
            c = color
        ellipse = get_gaussian_ellipsiod(
            gaussians.mean[i],
            cov=gaussians.cov[i],
            color=c,
            opacity=opacity,
            resolution=resolution,
        )
        data.append(ellipse)

    return data


def get_frustum_mesh(
    frustum: Frustums, opacity: float = 0.3, color: str = "#DC203C", resolution: int = 20
) -> go.Mesh3d:  # type: ignore
    """Get a plotly mesh for a single frustum.

    Args:
        frustum: Single frustum
        opacity: Opacity of the mesh. Defaults to 0.3.
        color: Color of the mesh. Defaults to "#DC203C".
        resolution: Resolution of the mesh. Defaults to 20.

    Returns:
        Plotly mesh
    """

    if frustum.ndim > 1:
        raise ValueError("Frustum must be a single Frustum object.")

    base_radius = torch.sqrt(frustum.pixel_area / torch.pi)
    f_radius = frustum.starts * base_radius
    b_radius = frustum.ends * base_radius

    x = torch.cat([torch.ones(resolution) * frustum.starts, torch.ones(resolution) * frustum.ends])
    pts = torch.linspace(0, 2 * torch.pi, resolution)

    y = torch.sin(pts)
    z = torch.cos(pts)
    y = torch.cat([y * f_radius, y * b_radius])
    z = torch.cat([z * f_radius, z * b_radius])

    pts = torch.stack([x, y, z], dim=-1)

    forward = frustum.directions
    up = F.normalize(torch.cross(forward, torch.tensor([0.0, 0, 1])), dim=-1)  # type: ignore
    right = F.normalize(torch.cross(forward, up), dim=-1)  # type: ignore
    rotation = torch.stack([forward, up, right], dim=1)
    pts = torch.einsum("kj,ij->ki", pts, rotation)

    pts += frustum.origins
    return go.Mesh3d(
        x=pts[..., 0],
        y=pts[..., 1],
        z=pts[..., 2],
        opacity=opacity,
        alphahull=0,
        color=color,
        flatshading=True,
        name="Frustums",
    )


def get_frustums_mesh_list(
    frustums: Frustums, opacity: float = 1.0, color: str = "random", resolution: int = 20
) -> List[go.Mesh3d]:  # type: ignore
    """Get a list of plotly meshes for a list of frustums.

    Args:
        frustums: List of frustums
        opacity: Opacity of the mesh. Defaults to 0.3.
        color: Color of the mesh. Defaults to "random".
        resolution: Resolution of the mesh. Defaults to 20.

    Returns:
        List of plotly meshes
    """
    data = []
    for i, frustum in enumerate(frustums.flatten()):  # type: ignore
        if color == "random":
            c = get_random_color(idx=i)
        else:
            c = color
        data.append(get_frustum_mesh(frustum, opacity=opacity, color=c, resolution=resolution))
    return data


def get_frustum_points(
    frustum: Frustums, opacity: float = 1.0, color: str = "forestgreen", size: float = 5
) -> go.Scatter3d:  # type: ignore
    """Get a set plotly points for frustums centers.

    Args:
        frustum: Frustums to visualize.
        opacity: Opacity of the points. Defaults to 0.3.
        color: Color of the poinst. Defaults to "forestgreen".
        size: Size of points. Defaults to 10.

    Returns:
        Plotly points
    """

    frustum = frustum.flatten()
    pts = frustum.get_positions()


    return go.Scatter3d(
        x=pts[..., 0],
        y=pts[..., 1],
        z=pts[..., 2],
        mode="markers",
        marker=dict(
            size=size,
            color=color,
            opacity=opacity,
        ),
        name="Frustums -> Positions",
    )


def get_ray_bundle_lines(
    ray_bundle: RayBundle, length: float = 1.0, color: str = "#DC203C", width: float = 1
) -> go.Scatter3d:  # type: ignore
    """Get a plotly line for a ray bundle.

    Args:
        ray_bundle: Ray bundle
        length: Length of the line. Defaults to 1.0.
        color: Color of the line.
        width: Width of the line. Defaults to 1.

    Returns:
        Plotly lines
    """

    origins = ray_bundle.origins.view(-1, 3)
    directions = ray_bundle.directions.view(-1, 3)

    lines = torch.empty((origins.shape[0] * 2, 3))
    lines[0::2] = origins
    lines[1::2] = origins + directions * length
    return go.Scatter3d(
        x=lines[..., 0],
        y=lines[..., 1],
        z=lines[..., 2],
        mode="lines",
        name="Ray Bundle",
        line=dict(color=color, width=width),
    )


def vis_camera_rays(cameras: Cameras) -> go.Figure:  # type: ignore
    """Visualize camera rays.

    Args:
        camera: Camera to visualize.

    Returns:
        Plotly lines
    """

    coords = cameras.get_image_coords()
    coords[..., 0] /= cameras.image_height[0]  # All the cameras have the same image height for now
    coords[..., 1] /= cameras.image_width[0]  # All the cameras have the same image width for now
    coords = torch.cat([coords, torch.ones((*coords.shape[:-1], 1))], dim=-1)

    ray_bundle = cameras.generate_rays(camera_indices=0)

    origins = ray_bundle.origins.view(-1, 3)
    directions = ray_bundle.directions.view(-1, 3)
    coords = coords.view(-1, 3)

    lines = torch.empty((origins.shape[0] * 2, 3))
    lines[0::2] = origins
    lines[1::2] = origins + directions

    colors = torch.empty((coords.shape[0] * 2, 3))
    colors[0::2] = coords
    colors[1::2] = coords

    fig = go.Figure(
        data=go.Scatter3d(
            x=lines[:, 0],
            y=lines[:, 2],
            z=lines[:, 1],
            marker=dict(
                size=4,
                color=colors,
            ),
            line=dict(color="lightblue", width=1),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x", showspikes=False),
            yaxis=dict(title="z", showspikes=False),
            zaxis=dict(title="y", showspikes=False),
        ),
        margin=dict(r=0, b=10, l=0, t=10),
        hovermode=False,
    )

    return fig


def get_camera_frustums(cameras: Cameras):
    """Returns the camera frustums for the cameras that we are using.

    Args:
        cameras: The cameras that we want to plot.

    Returns:
        A plotly scatter that can be plotted.
    """
    for camera_idx in range(cameras.size):
        json_ = cameras.to_json(camera_idx=camera_idx)
        print(json_)
    raise NotImplementedError

def vis_3d_points(pts):

    if isinstance(pts,torch.Tensor):
        pts = pts.detach().cpu().numpy()
    fig = go.Figure()
    for i in range(pts.shape[0]):
        fig.add_trace(
                go.Scatter3d(x = pts[i,list(range(0,245,3)),0],
                             y = pts[i,list(range(0,245,3)),1],
                             z = pts[i,list(range(0,245,3)),2],
                            mode="markers+lines",
                            marker=dict(
                                size=2,
                                color="green",
                                opacity=1,
                            ),
                           line=dict(
                              color="blue"
                           ),
                    name=f"Frustums -> Positions{i}",
                )
            )

    fig.write_html("3d_pts.html")
    return

def draw_bbx(vertices,filename = None):
    vertices = vertices.squeeze()
    num_bbx = vertices.shape[0]
    fig = go.Figure()
    # self.max_boarder = vertices.max()
    ## i, j and k give the vertices of triangles (代表了连接顺序) 发现0123 0257 4567是一个面
    ## 每次取出 i j,k 的元素组成一个三角形，因此012 123 两个三角形组成了长方体的一个面
    i = [0, 3, 2, 5, 4, 5, 1, 1,3,3,1,1]
    j = [1, 2, 5, 0, 5, 6, 6, 6,7,7,5,5]
    k = [2, 1, 7, 2, 6, 7, 4, 3,2,6,0,4]
    for idx in range(num_bbx):
        fig.add_trace(
            go.Mesh3d(
            x = vertices[idx,:,0],
            y = vertices[idx,:,1],
            z = vertices[idx,:,2],
            i = i,
            j = j,
            k = k,
            opacity=0.1,
            color ='blue',
            name = f"bbx_{idx}"
            )
        )
    fig.write_html("Vis_bbx.html")

def draw_rays_near_far( rays_o, nears, fars, output_name="vis_intersection.html", indices=None):
    if len(indices) < 5:
        return
    idx = indices.squeeze()[:5]
    fig = go.Figure()
    pts_nears = nears[idx]
    pts_fars = fars[idx]
    # rays_o = rays_o[idx]
    # pts = torch.stack((rays_o,pts_nears,pts_fars),dim=1)
    pts = torch.stack((pts_nears, pts_fars), dim=1)
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()

    for i in range(len(idx)):
        fig.add_trace(
            go.Scatter3d(x=pts[i, :, 0], y=pts[i, :, 1], z=pts[i, :, 2],
                         mode='markers+lines',
                         marker=dict(size=2)
                         )
        )