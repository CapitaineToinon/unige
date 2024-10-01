#let author = "Antoine Sutter"
#let email = "antoine.sutter@etu.unige.ch"
#let title = "TP1: Linear Algebra"

#set heading(numbering: "1.")
#set page(
  paper: "a4",
  header: context {
    if counter(page).get().first() > 1 [
      #author
      #h(1fr) #title
    ]
  },
  footer: context {
    if counter(page).get().first() > 1 [
      #set align(right)
      #set text(8pt)
      #counter(page).display(
        "1 of I",
        both: true,
      )
    ]
  },
)

#align(center)[
  #text(size: 20pt)[Université de Genève]

  #text(size: 18pt)[
    Analyse et Traitement de l'Information\
    14X026
  ]

  #let padding = 30pt
  #pad(top: padding, line(length: 100%))
  #pad(y: padding, text(size: 24pt, [*#title*]))
  #pad(bottom: padding, line(length: 100%))

  #author\
  #link("mailto:" + email)[#email]
]

#align(bottom + center)[
  #datetime.today().display("[month repr:long] [year]")
  #figure(image("unige_csd.png"))
]

#pagebreak()

#outline()

#pagebreak()

= Matrix

== Quadratic polynomial

Find a quadratic polynomial, say $f(x) = a x^2 + b x + c$, such that
$ f(1) = -5, f(2) = 1, f(3) = 11 $

First lets fill the values:

+ $f_1 = f(1) = a + b + c = -5$
+ $f_2 = f(2) = 4a + 2b + c = 1$
+ $f_3 = f(3) = 9a + 3b + c = 11$

First subscract $f_1$ from $f_2$ to get rid of $c$:

- $4a + 2b + c = 1$
- $a + b + c = -5$

That gives us equation $f_4$:

- $3a + b = 6$

Then subscract $f_2$ from $f_3$ to again get rid of $c$:

- $9a + 3b + c = 11$
- $4a + 2b + c = 1$

That gives us equation $f_5$:

- $5a + b = 10$

Now let's subscract $f_4$ from $f_5$ to solve $a$:

- $5a + b = 10$
- $3a + b = 6$

Which gives:

- $2a = 4$
- $a = 2$

Now that we know that $a = 2$ we can solve $b$ by injecting $a$ into $f_4$:

- $3a + b = 6$
- $6 + b = 6$
- $b = 0$

Finally we can inject $a$ and $b$ into $f_1$ to solve $c$:

- $a + b + c = -5$
- $2 + c = -5$
- $c = -7$

Therefore the quatdratic polynomial is the following:

- $f(x) = 2 x^2 - 7$

#pagebreak()

== Linear equation system

Let $a$, $b$ be some fixed parameters. Solve the system of linear equations

$
  cases(
    x + a y = 2,
    b x + 2 y = 3
  )
$

First I solve for $x$:

$
  x + a y = 2 \
  x = 2 - a y
$

Then substitude x into the 2nd equation

$
  b x + 2 y = 3 \
  b(2 - a y) + 2y = 3 \
  2b - a b y + 2y = 3 \
  2b + 2y - a b y = 3
$

Now move all the $y$ on one side

$ 2y - a b y = 3 - 2b $

Then factor $y$ out and isolate $y$

$
  y (2 - a b) = 3 - 2b \
  y = (3 - 2b) / (2 - a b)
$

We can now take back the equation were we isolated $x$ and inject $y$ into it

$
  x = 2 - a y \
  x = 2 - a (3 - 2b) / (2 - a b) \
  x = 2 - (a(3 - 2b)) / (2 - a b) \
  x = (2(2 - a b)) / (2 - a b) - (a(3 - 2b)) / (2 - a b) \
  x = (2(2 - a b) - a(3 - 2b)) / (2 - a b) \
  x = (4 - 2 a b - 3a + 2 a b) / (2 - a b) \
  x = (4 - 3a) / (2 - a b)
$

We now have a solution for both $x$ and $y$ with the condition that $2 - a b != 0$

$
  x = (4 - 3a) / (2 - a b) \
  y = (3 - 2b) / (2 - a b)
$

#pagebreak()

== Matrix inverse

Find the inverse of the following matrix, if it exists

$
  A = mat(
    1, 2, 3;
    0, 1, 4;
    5, 6, 0;
  )
$

First let's assign a letter to each value in $A$

$
  #let a = 1
  #let b = 2
  #let c = 3
  #let d = 0
  #let e = 1
  #let f = 4
  #let g = 5
  #let h = 6
  #let i = 0

  a = #a, b = #b, c = #c \
  d = #d, e = #e, f = #f \
  g = #g, h = #h, i = #i \
$

From there we can calulate the deteminant and check if it's not 0. Otherwise it has no inverse solution.

$
  det(A) = a(e i - f h) - b(d i - f g) + c(d h - e g) \
  det(A) = #a (#e * #i - #f * #h) - #b (#d * #i - #f * #g) + #c (#d * #h - #e * #g) \
  det(A) = (#a * #(e * i - f * h)) - (#b * #(d * i - f * g)) + (#c * #(d * h - e * g)) \
  det(A) = (#(a * (e * i - f * h))) - (#(b * (d * i - f * g))) + (#(c * (d * h - e * g))) \
  det(A) = #(a * (e * i - f * h) - (b * (d * i - f * g)) + (c * (d * h - e * g)))
$

Because the deteminant isn't 0 we know there is a solution. We can now calculate the cofactor matrix of $A$. The cofactor $C i j$ is calculated by eliminating the $i$-th row and $j$-th column, then calculating the determinant of the resulting 2x2 matrix, applying a sign change based on position.

$
  C 11: det(1*0 - 4*6) = -24 \
  C 12: -det(0*0 - 4*5) = 20 \
  C 13: det(0*6 - 1*5) = -5 \
  C 21: -det(2*0 - 3*6) = 18 \
  C 22: det(1*0 - 3*5) = -15 \
  C 23: -det(1*4 - 2*5) = 4 \
  C 31: det(2*4 - 3*1) = 5 \
  C 32: -det(1*4 - 3*0) = -4 \
  C 33: det(1*1 - 2*0) = 1 \
$

This gives us the $"Cof"(A)$ matrix:

$
  "Cof"(A) = mat(
    -24, 20, -5;
    18, -15, 4;
    5, -4, 1;
  )
$

We can then transpose the $"Cof"(A)$:

$
  "Adj"(A) = mat(
    -24, 18, 5;
    20, -15, -4;
    -5, 4, 1;
  )
$

And that gives us thus the inverse of $A$.

#pagebreak()

== Matrix equation for $X$

Solve the following matrix equation for $X$:

$
  A X = B, "where" A = mat(1, 3; 2, 5), B = mat(7, 8; 9, 10)
$

Let's start with the determinant of $A$:

$
  det(A) = (1 * 5) - (3 * 2) = 5 - 6 = -1
$

From there we can get $A^(-1)$

$
  A^(-1) = 1 / det(A) * mat(d, -b; -c, a)
$

With values filled we get the following

$
  A^(-1) = 1 / (-1) * mat(5, -3; -2, 1)\
  A^(-1) = mat(-5, 3; 2, -1)
$

Now we need to multiply $A^(-1)$ by $B$ to find $X$

$
  X = mat(-5, 3; 2, -1) * mat(7, 8; 9, 10)\
$

$
  X[0][0] = (-5 * 7) + (3 * 9) = -35 + 27 = -8 \
  X[0][1] = (-5 * 8) + (3 * 10) = -40 + 30 = -10 \
  X[1][0] = (2 * 7) + (-1 * 9) = 14 - 9 = 5 \
  X[1][1] = (2 * 8) + (-1 * 10) = 16 - 10 = 6 \
$

Which means we have our solution:

$
  X = mat(
    -8, -10;
    5, 6;
  )
$

Because we can verify that $A X = B$

$
  mat(1, 3; 2, 5) mat(-8, -10;5, 6) = mat(7, 8; 9, 10)
$

#pagebreak()

= Math concepts behind code

Firsly before running the `some_script.py` file, I added the following code to supress the deprecation warnings as the output is really noisy otherwise. This really shouldn't be a thing but the point of this exercice is to explain the code, not fix it.

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

Example output with the warnings:

```
some_script.py:10: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  print('START - %s'%datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]);
START - 2024-10-01 16:11:56.612
Result = 0.0
some_script.py:53: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  print('END - %s\n'%datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]);
END - 2024-10-01 16:11:56.756
```

And without the warnings:

```
START - 2024-10-01 16:12:33.537
Result = 0.0
END - 2024-10-01 16:12:33.584
```

#pagebreak()

== `def project_on_first(u, v)`

Like the same suggests, this fucking allows you to project a verctor into another vector, in this case project $v$ onto $u$. This returns a new vector $w$ which can be thought of as the shadow of $u$.

This consists of a few steps

+ dot product between $u$ and $v$
+ dot product between $u$ and $u$
+ Scalar product using the two dot products from above
+ Projecting the vector

A dot product between two matrices consist in multiplying each cell with the corresponding cell of the other vectoc and summing all the results. Consider $u$ and $v$:

$ u = mat(3, 4), v = mat(1, 2) $

Then a dot product of $u #math.dot v$ is done as follow:

$ u #math.dot v = mat(3, 4) #math.dot mat(1, 2) = 3 * 1 + 4 * 2 = 11 $

Then doing the dot project $u #math.dot u$ is done as follow:

$ u #math.dot v = mat(3, 4) #math.dot mat(3, 4) = 3 * 3 + 4 * 4 = 25 $

After that, we can do the scalar product, which consist of the following:

$ "scalar" = (u #math.dot v) / (u #math.dot u) -> 11 / 25 $

The projection is then a multiplication of the result above with the $u$ vector.

$ w = 11 / 25 * mat(3, 4) = mat(33 / 25, 44 / 25) #math.approx mat(1.32, 1.76) $

In summary, the function projects vector $v$ onto vector $u$, resulting in a new vector that represents the part of $v$ that is aligned with $u$.

#pagebreak()

== Python function into math

```python
# import numpy library as np in order to work on tensors
import numpy as np
# create 2 random vectors of length 7
u = np.random.randn(7)
v = np.random.randn(7)
# perform some computations
r=0
for ui, vi in zip(u, v):
  r += ui * vi
```

The code above is an implementation of a dot product between two matrices. Mathematically this can be expressed as such:

$ u #math.dot v $

If we want to be pedentic, we could rewite the last tree lines into one as such:

#align(center)[
  ```python
  r = sum([ui * vi for ui, vi in zip(u, v)])
  ```
]


But of course would should use `numpy` for this instead:

#align(center)[
  ```python
  r = np.dot(u, v)
  ```
]

#pagebreak()

== Orthogonal vector

Complete the code to construct a vector $v$ orthogonal to the vector $u$ and of the same norm. Comment each line of your code.

```python
import numpy as np

# create 2 random vectors of length 7
u = np.random.randn(7)
v = np.random.randn(7)

# Perform some computations
r = np.dot(u, v)

# Creating a vector orthogonal to u and of the same norm
# Step 1: Generate a random vector
w = np.random.randn(7)

# Step 2: Make w orthogonal to u by subtracting its projection on u
proj_u_w = (np.dot(w, u) / np.dot(u, u)) * u
v_orthogonal = w - proj_u_w

# Step 3: Normalize v_orthogonal to make it of the same norm as the original v
v_orthogonal = v_orthogonal / np.linalg.norm(v_orthogonal) * np.linalg.norm(v)

# Check orthogonality and norm
dot_product = np.dot(u, v_orthogonal)
norm_v_orthogonal = np.linalg.norm(v_orthogonal)

print("Dot product (should be close to 0):", dot_product)
print("Norm of v (original):", np.linalg.norm(v))
print("Norm of v (orthogonal):", norm_v_orthogonal)
```

Example output from a few runs I've done:

```
Dot product (should be close to 0): 0.0
Norm of v (original): 3.692870668658719
Norm of v (orthogonal): 3.6928706686587187
```

#pagebreak()

== Cosine of vectors

The cosine between two vectors $u$ and $v$ is defined as such:

$ cos #math.alpha = (u #math.dot v) / (#math.norm("u") #math.dot #math.norm("v")) $

In order to do that in python, we can use the following code. It's important to check that neither the norm of $u$ or $v$ is zero as the computation would be impossible otherwise.

```python
def cosine_similarity(u: np.array, v: np.array) -> np.float64:
    # Get the norm for u and v
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    # Prevent a divide by 0 error
    if norm_u == 0 or norm_v == 0:
        raise ValueError("cannot use vectors with norm 0")

    # do the actual computation
    return np.dot(u, v) / (norm_u * norm_v)

u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
cos_theta = cosine_similarity(u, v)
print(f'Cosine of the angle between u and v: {cos_theta}')
```

In this case, the angle is `0.9746318461970762`.
