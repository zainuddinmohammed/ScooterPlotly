# 🛴 Scooter Plotly Library

**Scooter Plotly** is a powerful and customizable 3D visualization library built on top of [Plotly](https://plotly.com/python/), designed for visualizing coordinate geometry, vector math, surfaces, parametric equations, and animated 3D models in Python.

---

## 🚀 Features

- 📐 3D coordinate axes with labeled tick marks and optional gridlines
- 🧭 Position, directional, and appended vectors with arrowheads
- 🌐 Implicit and parametric surfaces (including spheres and curves)
- 🌀 Symbolic expression support using SymPy
- 🎥 3D `.obj` model loading and animation with rotation and translation
- 🖼️ Beautiful dark-themed layouts
- 🔧 Configurable view bounds, axis ranges, and colors

---

## 📦 Installation

This library is not yet on PyPI. To use it:

1. Clone or copy the `scooter_plotly.py` file to your project directory.
2. Install the required packages:

```bash
pip install numpy plotly sympy trimesh
```

---

## 🧪 Example Usage

### Plotting a position vector

```python
from ScooterPlotly import *

vec = PositionVector([5, 4, 3], name='r')
axes_set = axes(step=2)

CreateFigure([*axes_set, *vec], "Position Vector Example")
```

### Rendering a sphere

```python
from ScooterPlotly import *

L = [
    modelSphere(h=0, k=0, l=0, r=5),
    axes(step=2, xy=True)
]

CreateFigure(L, "Sphere Centered at Origin")
```

---

## 🔧 Configuration

Use `GridSpace(m, M)` to set the global coordinate space:
```python
GridSpace(-10, 10)  # sets x, y, z ranges from -10 to 10
```

Customize axis ranges:
```python
GridSpace(-10, 10, xmin=-5, xmax=15)  # sets y, z ranges from -10 to 10, x range to -5, 15
```

Toggle Plotly's default axis/grid visibility with:
```python
GridSpace(-10, 10, default_axes=False)
```

---

## 🧰 Core Functions

### Axes

- `axes(step=None, xy=False)` – Generates all three labeled axes, step indicates ticks of each axis
- `axes(step=None, xy=True)` – Generates xy-grid to imitate Desmos 3D UI
- `x_axis(color, name, step)` – Individual axis generator (also `y_axis`, `z_axis`)

### Vectors

- `PositionVector(p, name, color)` – From origin to point
- `Vector(p1, p2, name, color)` – From one point to another
- `AppendedVector(p1, p2, ...)` – Vector chained after another
- `Uvec(v)` – Unit vector
- `Projection(u, v)` – Projection of vector `u` onto `v`

### Points

- `Point(p, name)` – Plot a labeled point
- `PointVectors(p, ...)` – XYZ component vectors to a point

### Surfaces & Curves

- `modelSphere(h, k, l, r)` – Parametric sphere
- `model3DCartesFunction(expr)` – Implicit surface from expression
- `model3DParamFunction(x_expr, y_expr, z_expr)` – Parametric 3D curve

### 3D Object Support

- `loadObject(folder_path, direction, position, ...)` – Load `.obj` model
- `loadObjectAnimation(path, direction, pos1, pos2, ...)` – Animate object movement

---

## 🖼️ Rendering

- `CreateFigure(models, title)` – Generates and opens the plot in your browser
- `breakdown(models)` – Flattens nested lists of models (usually not needed directly)

---

## 🎨 Styling & Appearance

- Background: Black theme
- Axes: White lines with arrowhead cones
- Color maps: `'Plasma'`, `'Viridis'`, custom gradients like `'fire'` and `'colorscale'`
- Fonts: Arial, white-colored labels
- Aspect ratio: Cube by default

---

## 🧠 Dependencies

- `numpy` – Vector/matrix math
- `plotly` – 3D interactive graphing
- `sympy` – Symbolic math
- `trimesh` – Mesh manipulation and .obj import
- `webbrowser` / `os` – To open plots locally

---

## 📌 Notes

- HTML output (`plot.html`) is overwritten each time you call `CreateFigure()`
- Cone vectors scale with grid size automatically
- Symbolic expressions must use `sympy.symbols('x y z t')`

---

## 📎 License

This project is released under the **MIT License**.

---

## ✍️ Author

Created by **Zainuddin Mohammed**  
Visualization powered by 🛴 Scooter and 🧠 math.

---

## 🔮 Future Additions

- GUI-based interactive plot editor (like Desmos 3D)
- More available color schemes for UI and graphing
- Wider range of parameters for lines/surfaces, models, and 

---

![The Scooter in Question](https://i.imgur.com/CT8sDui.gif)
