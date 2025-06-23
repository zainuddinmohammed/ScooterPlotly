import numpy as np
from sympy import *
import plotly.graph_objs as go
import trimesh
import webbrowser
import os

# VARIABLES AND FUNCTIONS SETUP

config = {
    'MIN': -20,
    'MAX': 20,
    'MINX': -20, 'MAXX': 20,
    'MINY': -20, 'MAXY': 20,
    'MINZ': -20, 'MAXZ': 20,

    'DEFAULT_AXES': True,

    'xy_gridlines': False,

    'light': False,
}

x, y, z, t = symbols('x y z t')

colorscale = [
    [0, 'blue'],
    [1, 'indigo'],
]
fire = [
    [0, 'white'],
    [1, 'orange'],
]


# CREATE AXES

def x_axis(color, name, step=None):
    line = go.Scatter3d(
        x=[config['MINX'], config['MAXX']],
        y=[0, 0],
        z=[0, 0],
        mode="lines+text",
        text=["", name],
        textposition='top center',
        textfont=dict(
            color=color,
            family="Arial",
            size=20,
        ),
        marker=dict(size=5, color=color),
        line=dict(width=3, color=color),
        name=name,
        showlegend=False,
        hoverinfo='skip',
    )
    cone1 = go.Cone(
        x=[config['MAXX']],
        y=[0],
        z=[0],
        u=[config['MAXX']],
        v=[0],
        w=[0],
        sizemode='absolute',
        sizeref=(config['MAX'] - config['MIN']) / 40,
        anchor='tip',
        colorscale=[[0, color], [1, color]],
        showscale=False,
        hoverinfo='skip'
    )
    cone2 = go.Cone(
        x=[config['MINX']],
        y=[0],
        z=[0],
        u=[config['MINX']],
        v=[0],
        w=[0],
        sizemode='absolute',
        sizeref=(config['MAX'] - config['MIN']) / 40,
        anchor='tip',
        colorscale=[[0, color], [1, color]],
        showscale=False,
        hoverinfo='skip'
    )
    tickList = []
    if step is not None:
        numTicksPos = abs(int(config['MAXX'] / step)) + 1
        numTicksNeg = abs(int(config['MINX'] / step)) + 1

        for i in range(numTicksPos):
            if config['xy_gridlines']:
                gridline_x = go.Scatter3d(
                    x=[config['MINX'], config['MAXX']],
                    y=[i * step, i * step],
                    z=[0, 0],
                    mode="lines",
                    line=dict(width=1, color='#8c8c8c'),
                    showlegend=False,
                    hoverinfo='skip'
                )
                tickList.append(gridline_x)
            tick = go.Scatter3d(
                x=[0, i * step],
                y=[0, 0],
                z=[0, 0],
                mode="text+markers",
                marker=dict(
                    color=color,
                    size=3
                ),
                text=["", str(i * step)],
                textposition='top right',
                textfont=dict(
                    color=color,
                    family="Arial",
                    size=12,
                ),
                showlegend=False,
                hoverinfo='skip'

            )
            tickList.append(tick)
        for i in range(numTicksNeg):
            if config['xy_gridlines']:
                gridline_x = go.Scatter3d(
                    x=[config['MINX'], config['MAXX']],
                    y=[-i * step, -i * step],
                    z=[0, 0],
                    mode="lines",
                    line=dict(width=1, color='#8c8c8c'),
                    showlegend=False,
                    hoverinfo='skip'
                )
                tickList.append(gridline_x)
            tick = go.Scatter3d(
                x=[0, -i * step],
                y=[0, 0],
                z=[0, 0],
                mode="text+markers",
                marker=dict(
                    color=color,
                    size=3
                ),
                text=["", str(-i * step)],
                textposition='top right',
                textfont=dict(
                    color=color,
                    family="Arial",
                    size=12,
                ),
                showlegend=False,
                hoverinfo='skip'

            )
            tickList.append(tick)

    return [line, cone1, cone2] + tickList


def y_axis(color, name, step=None):
    line = go.Scatter3d(
        x=[0, 0],
        y=[config['MINY'], config['MAXY']],
        z=[0, 0],
        mode="lines+text",
        text=["", name],
        textposition='top center',
        textfont=dict(
            color=color,
            family="Arial",
            size=20,
        ),
        marker=dict(size=5, color=color),
        line=dict(width=3, color=color),
        name=name,
        showlegend=False,
        hoverinfo='skip'
    )
    cone1 = go.Cone(
        x=[0],
        y=[config['MAXY']],
        z=[0],
        u=[0],
        v=[config['MAXY']],
        w=[0],
        sizemode='absolute',
        sizeref=(config['MAX'] - config['MIN']) / 40,
        anchor='tip',
        colorscale=[[0, color], [1, color]],
        showscale=False,
        hoverinfo='skip'
    )
    cone2 = go.Cone(
        x=[0],
        y=[config['MINY']],
        z=[0],
        u=[0],
        v=[config['MINY']],
        w=[0],
        sizemode='absolute',
        sizeref=(config['MAX'] - config['MIN']) / 40,
        anchor='tip',
        colorscale=[[0, color], [1, color]],
        showscale=False,
        hoverinfo='skip'
    )
    tickList = []
    if step is not None:
        numTicksPos = abs(int(config['MAXY'] / step)) + 1
        numTicksNeg = abs(int(config['MINY'] / step)) + 1

        for i in range(numTicksPos):
            if config['xy_gridlines']:
                gridline_y = go.Scatter3d(
                    x=[i * step, i * step],
                    y=[config['MINY'], config['MAXY']],
                    z=[0, 0],
                    mode="lines",
                    line=dict(width=1, color='#8c8c8c'),
                    showlegend=False,
                    hoverinfo='skip'
                )
                tickList.append(gridline_y)
            tick = go.Scatter3d(
                x=[0, 0],
                y=[0, i * step],
                z=[0, 0],
                mode="text+markers",
                marker=dict(
                    color=color,
                    size=3
                ),
                text=["", str(i * step)],
                textposition='top right',
                textfont=dict(
                    color=color,
                    family="Arial",
                    size=12,
                ),
                showlegend=False,
                hoverinfo='skip'

            )
            tickList.append(tick)
        for i in range(numTicksNeg):
            if config['xy_gridlines']:
                gridline_y = go.Scatter3d(
                    x=[-i * step, -i * step],
                    y=[config['MINY'], config['MAXY']],
                    z=[0, 0],
                    mode="lines",
                    line=dict(width=1, color='#8c8c8c'),
                    showlegend=False,
                    hoverinfo='skip'
                )
                tickList.append(gridline_y)
            tick = go.Scatter3d(
                x=[0, 0],
                y=[0, -i * step],
                z=[0, 0],
                mode="text+markers",
                marker=dict(
                    color=color,
                    size=3
                ),
                text=["", str(-i * step)],
                textposition='top right',
                textfont=dict(
                    color=color,
                    family="Arial",
                    size=12,
                ),
                showlegend=False,
                hoverinfo='skip'

            )
            tickList.append(tick)

    return [line, cone1, cone2] + tickList


def z_axis(color, name, step=None):
    line = go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[config['MINZ'], config['MAXZ']],
        mode="lines+text",
        text=["", name],
        textposition='top center',
        textfont=dict(
            color=color,
            family="Arial",
            size=20,
        ),
        marker=dict(size=5, color=color),
        line=dict(width=3, color=color),
        name=name,
        showlegend=False,
        hoverinfo='skip'

    )
    cone1 = go.Cone(
        x=[0],
        y=[0],
        z=[config['MAXZ']],
        u=[0],
        v=[0],
        w=[config['MAXZ']],
        sizemode='absolute',
        sizeref=(config['MAX'] - config['MIN']) / 40,
        anchor='tip',
        colorscale=[[0, color], [1, color]],
        showscale=False,
        hoverinfo='skip'

    )
    cone2 = go.Cone(
        x=[0],
        y=[0],
        z=[config['MINZ']],
        u=[0],
        v=[0],
        w=[config['MINZ']],
        sizemode='absolute',
        sizeref=(config['MAX'] - config['MIN']) / 40,
        anchor='tip',
        colorscale=[[0, color], [1, color]],
        showscale=False,
        hoverinfo='skip'

    )
    tickList = []
    if step is not None:
        numTicksPos = abs(int(config['MAXZ'] / step)) + 1
        numTicksNeg = abs(int(config['MINZ'] / step)) + 1

        for i in range(numTicksPos):
            tick = go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[0, i*step],
                mode="text+markers",
                marker=dict(
                    color=color,
                    size=3
                ),
                text=["", str(i*step)],
                textposition='middle right',
                textfont=dict(
                    color=color,
                    family="Arial",
                    size=12,
                ),
                showlegend=False,
                hoverinfo='skip'
            )
            tickList.append(tick)
        for i in range(numTicksNeg):
            tick = go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[0, -i*step],
                mode="text+markers",
                marker=dict(
                    color=color,
                    size=3
                ),
                text=["", str(-i*step)],
                textposition='middle right',
                textfont=dict(
                    color=color,
                    family="Arial",
                    size=12,
                ),
                showlegend=False,
                hoverinfo='skip'

            )
            tickList.append(tick)

    return [line, cone1, cone2] + tickList


def axes(color='white', step=None, xy=False, x=True, y=True, z=True):
    if xy:
        config['xy_gridlines'] = True
    else:
        config['xy_gridlines'] = False
    ret = []
    if x:
        ret.append(x_axis(color, 'x', step=step))
    if y:
        ret.append(y_axis(color, 'y', step=step))
    if z:
        ret.append(z_axis(color, 'z', step=step))
    return ret


# VECTOR FUNCTIONS

# Above with arrow
def PositionVector(p, color='red', name='', legend=True):
    line = go.Scatter3d(
        x=[0, p[0]],
        y=[0, p[1]],
        z=[0, p[2]],
        mode="lines+text",
        text=["", name],
        textposition='top center',
        textfont=dict(
            color=color,
            family="Arial",
            size=20,
        ),
        marker=dict(size=5, color=color),
        line=dict(width=6, color=color),
        name=name,
        showlegend=legend
    )
    cone = go.Cone(
        x=[p[0]],
        y=[p[1]],
        z=[p[2]],
        u=[p[0]],
        v=[p[1]],
        w=[p[2]],
        sizemode='absolute',
        sizeref=(config['MAX'] - config['MIN']) / 40,
        anchor='tip',
        colorscale=[[0, color], [1, color]],
        showscale=False
    )
    return [line, cone]


# Above with arrow
def Vector(p1, p2, color, name, legend=True):
    line = go.Scatter3d(
        x=[p1[0], p2[0]],
        y=[p1[1], p2[1]],
        z=[p1[2], p2[2]],
        mode="lines+text",
        text=["", name],
        textposition='top center',
        textfont=dict(
            color=color,
            family="Arial",
            size=20,
        ),
        marker=dict(size=5, color=color),
        line=dict(width=6, color=color),
        name=name,
        showlegend=legend
    )
    cone = go.Cone(
        x=[p2[0]],
        y=[p2[1]],
        z=[p2[2]],
        u=[p2[0] - p1[0]],
        v=[p2[1] - p1[1]],
        w=[p2[2] - p1[2]],
        sizemode='absolute',
        sizeref=(config['MAX'] - config['MIN']) / 40,
        anchor='tip',
        colorscale=[[0, color], [1, color]],
        showscale=False
    )
    return [line, cone]


# Models a vector appended to another vector
def AppendedVector(p1, p2, color='red', name='', legend=True):
    line = go.Scatter3d(
        x=[p1[0], p1[0] + p2[0]],
        y=[p1[1], p1[1] + p2[1]],
        z=[p1[2], p1[2] + p2[2]],
        mode="lines+text",
        text=["", name],
        textposition='top center',
        textfont=dict(
            color=color,
            family="Arial",
            size=20,
        ),
        marker=dict(size=5, color=color),
        line=dict(width=6, color=color),
        name=name,
        showlegend=legend,
    )
    cone = go.Cone(
        x=[p1[0] + p2[0]],
        y=[p1[1] + p2[1]],
        z=[p1[2] + p2[2]],
        u=[p2[0]],
        v=[p2[1]],
        w=[p2[2]],
        sizemode='absolute',
        sizeref=(config['MAX'] - config['MIN']) / 40,
        anchor='tip',
        colorscale=[[0, color], [1, color]],
        showscale=False
    )
    return [line, cone]

# Returns a unit vector
def Uvec(v):
    mag = np.linalg.norm(v)
    return v / mag


# Finds vector projection of u onto v
def Projection(u, v):
    return np.dot(u, v) / np.linalg.norm(v) ** 2 * v


# Models a point
def Point(p, color='red', name=''):
    return go.Scatter3d(
        x=[p[0]],
        y=[p[1]],
        z=[p[2]],
        mode="markers+text",
        text=[name],  # ← this is what shows the label
        textposition="top center",  # ← optional: control label position
        textfont=dict(color=color),
        marker=dict(size=5, color=color),
        name=name
    )


# Returns a list of vectors that map the x-y-z directions from the origin to the specified point as well as the point
def PointVectors(p, color='red', name=''):
    return [
        Point(p, color, name),
        PositionVector(np.array([p[0], 0, 0]), color, '', legend=False),
        Vector(np.array([p[0], 0, 0]), np.array([p[0], p[1], 0]), color, '', legend=False),
        Vector(np.array([p[0], p[1], 0]), np.array([p[0], p[1], p[2]]), color, '', legend=False)
    ]


# SURFACES FUNCTIONS

# Models a sphere with center = (h, k, l) and radius = r
def modelSphere(h, k, l, r):
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    u, v = np.meshgrid(u, v)

    # Parametric equations
    x = h + r * np.cos(u) * np.sin(v)
    y = k + r * np.sin(u) * np.sin(v)
    z = l + r * np.cos(v)

    # Create the surface
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=colorscale,
        showscale=False,
        opacity=0.8
    )


def model3DCartesFunction(f_expr, grid_size=70):
    # Define symbolic expression
    x, y, z = symbols('x, y, z')

    # Lambdify to convert to NumPy function
    f = lambdify((x, y, z), f_expr, 'numpy')

    # Create 3D grid
    x_vals = np.linspace(config['MIN'], config['MAX'], grid_size)
    y_vals = np.linspace(config['MIN'], config['MAX'], grid_size)
    z_vals = np.linspace(config['MIN'], config['MAX'], grid_size)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)

    # Evaluate function
    values = f(X, Y, Z)

    return go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=-0.05,  # ← wider band so it catches surface
        isomax=0.05,
        surface_count=1,
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=False,
        colorscale='Plasma',
        opacity=0.9
    )


def model3DParamFunction(x_expr, y_expr, z_expr, color='red', name='', t_min=0.0, t_max=100.0, samples=200):
    # Lambdify the expressions
    x_func = lambdify(t, x_expr, modules='numpy')
    y_func = lambdify(t, y_expr, modules='numpy')
    z_func = lambdify(t, z_expr, modules='numpy')

    # Generate t values and evaluate
    t_vals = np.linspace(t_min, t_max, samples)

    x_eval = x_func(t_vals)
    x_vals = np.full_like(t_vals, x_eval, dtype=np.float64) if np.isscalar(x_eval) else x_eval

    y_eval = y_func(t_vals)
    y_vals = np.full_like(t_vals, y_eval, dtype=np.float64) if np.isscalar(y_eval) else y_eval

    z_eval = z_func(t_vals)
    z_vals = np.full_like(t_vals, z_eval, dtype=np.float64) if np.isscalar(z_eval) else z_eval

    # Return a 3D line trace
    return go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='lines',
        line=dict(color=color, width=4),
        name=name
    )


# 3D MODELS FUNCTIONS

# Normalize a vector (unit vector) with some checks
def normalize(v):
    norm = np.linalg.norm(v)
    if np.isclose(norm, 0):
        raise ValueError("Cannot normalize a zero vector")
    return v / norm


# Rotation functions
def rotation_matrix_from_vectors(v1, v2):
    v1 = normalize(v1)
    v2 = normalize(v2)
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    if np.isclose(dot, 1.0):
        return np.eye(3)  # No rotation needed
    if np.isclose(dot, -1.0):
        # 180-degree flip: find a stable perpendicular axis
        perp = np.eye(3)[np.argmin(np.abs(v1))]
        axis = normalize(np.cross(v1, perp))
        return rotation_matrix_from_axis_angle(axis, np.pi)

    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])
    R = np.eye(3) + skew + (skew @ skew) * ((1 - dot) / (np.linalg.norm(cross) ** 2))
    return R


def rotation_matrix_from_axis_angle(axis, theta):
    axis = normalize(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def loadObject(folder_path, direction=np.array([1, 0, 0]), position=np.array([0, 0, 0]), color='lightblue', scale=1):
    # Auto-detect .obj file in the folder
    obj_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.obj')]
    if not obj_files:
        raise FileNotFoundError("No .obj file found in the folder.")
    obj_path = os.path.join(folder_path, obj_files[0])

    # Load the .obj file with material/texture support
    scene = trimesh.load(obj_path, force='scene')

    # If it's a Scene, combine all meshes
    if isinstance(scene, trimesh.Scene):
        mesh = scene.dump(concatenate=True)
    else:
        mesh = scene  # Already a Trimesh

    # Apply scale
    vertices = mesh.vertices * scale
    faces = mesh.faces

    # Define forward direction
    model_forward = np.array([-1, 0, 0])
    R = rotation_matrix_from_vectors(model_forward, direction)
    vertices = vertices @ R.T

    # Upright correction (180° flip)
    forward_after_rotation = normalize(direction)
    R_upright = rotation_matrix_from_axis_angle(forward_after_rotation, 3 * np.pi / 2)
    vertices = vertices @ R_upright.T
    vertices = vertices + position

    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    # Try to extract color from material
    if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
        face_color = mesh.visual.face_colors[0][:3]
        color = f'rgb({face_color[0]}, {face_color[1]}, {face_color[2]})'

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=z,  # Use z-height as color driver
        colorscale='Viridis',  # Try others: 'Cividis', 'Jet', etc.
        flatshading=True,
        opacity=1.0,
        showscale=False
    )


def loadObjectAnimation(path, direction, pos1, pos2, color, scale):
    # Load the object once
    scene = trimesh.load(path)
    mesh = scene.to_geometry()

    # Apply scale
    base_vertices = mesh.vertices * scale
    faces = mesh.faces

    # Define directions
    model_forward = np.array([-1, 0, 0])

    # Rotate to face desired direction
    R = rotation_matrix_from_vectors(model_forward, direction)
    base_vertices = base_vertices @ R.T

    # Correct upside-down (rotate 270° around forward vector)
    forward_after_rotation = normalize(direction)
    R_upright = rotation_matrix_from_axis_angle(forward_after_rotation, 3 * np.pi / 2)
    base_vertices = base_vertices @ R_upright.T

    # Store all frames
    frames = []
    steps = 10
    for t in np.linspace(0, 1, steps):
        position = pos1 + (pos2 * t)
        vertices = base_vertices + position

        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

        mesh3d = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=1.0,
            color=color,
            flatshading=True
        )
        frames.append(go.Frame(data=[mesh3d], name=str(t)))

    frames.insert(0, 'anim_obj_flag')

    return frames


# Breaks down the models list into models (as some are lists of models such as the cone functions)
def breakdown(models):
    L = []
    for model in models:
        if isinstance(model, list):
            if model[0] == "anim_obj_flag":  # Prepend the first animation frame
                model.pop(0)
                m = model[0].data[0]
                L.insert(0, m)
            else:
                for m in model:
                    if isinstance(m, list):
                        for axis in m:
                            L.append(axis)
                    else:
                        L.append(m)
        else:
            L.append(model)
    return L


# LAYOUT SETTINGS
def GridSpace(m, M, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, default_axes=True, light_mode=False):

    if default_axes:
        config['DEFAULT_AXES'] = True
    else:
        config['DEFAULT_AXES'] = False

    if light_mode:
        config['light'] = True
    else:
        config['light'] = False

    if xmin is None:
        xmin = m
    if xmax is None:
        xmax = M
    if ymin is None:
        ymin = m
    if ymax is None:
        ymax = M
    if zmin is None:
        zmin = m
    if zmax is None:
        zmax = M

    config['MIN'], config['MAX'] = m, M

    config['MINX'], config['MAXX'] = xmin, xmax
    config['MINY'], config['MAXY'] = ymin, ymax
    config['MINZ'], config['MAXZ'] = zmin, zmax


def CreateFigure(data, title, filename='plot.html'):
    fig = go.Figure(data=breakdown(data), layout=go.Layout(
        width=1690,
        height=880,
        scene=dict(
            xaxis=dict(
                range=[config['MINX']-1, config['MAXX']+1],
                showgrid=True if config['DEFAULT_AXES'] else False,
                showticklabels=True if config['DEFAULT_AXES'] else False,
                title='x' if config['DEFAULT_AXES'] else '',
                backgroundcolor='#FFFCF2' if config['light'] else 'black',
                gridcolor='#262626',
                zerolinecolor='#c9c9c9' if config['DEFAULT_AXES'] else ('#FFFCF2' if config['light'] else 'black'),
                color='black' if config['light'] else 'white',
            ),
            yaxis=dict(
                range=[config['MINY']-1, config['MAXY']+1],
                showgrid=True if config['DEFAULT_AXES'] else False,
                showticklabels=True if config['DEFAULT_AXES'] else False,
                title='y' if config['DEFAULT_AXES'] else '',
                backgroundcolor='#FFFCF2' if config['light'] else 'black',
                gridcolor='#262626',
                zerolinecolor='#c9c9c9' if config['DEFAULT_AXES'] else ('#FFFCF2' if config['light'] else 'black'),
                color='black' if config['light'] else 'white',
            ),
            zaxis=dict(
                range=[config['MINZ']-1, config['MAXZ']+1],
                showgrid=True if config['DEFAULT_AXES'] else False,
                showticklabels=True if config['DEFAULT_AXES'] else False,
                title='z' if config['DEFAULT_AXES'] else '',
                backgroundcolor='#FFFCF2' if config['light'] else 'black',
                gridcolor='#262626',
                zerolinecolor='#c9c9c9' if config['DEFAULT_AXES'] else ('#FFFCF2' if config['light'] else 'black'),
                color='black' if config['light'] else 'white',
            ),
            aspectmode='cube'
        ),
        paper_bgcolor='#FFFCF2' if config['light'] else 'black',
        font=dict(family='Arial', color='white'),
        title=title,
        margin=dict(l=0, r=0, t=50, b=0),
    ))

    fig.write_html(filename)

    webbrowser.open('file://' + os.path.realpath(filename))
