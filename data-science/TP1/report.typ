#import "@local/unige:0.1.0": *

#show: doc => conf(
  title: "TP1: Linear Algebra",
  header: "Université de Genève",
  subheader: [
    Analyse et Traitement de l'Information\
    14X026
  ],
  authors: (
    (
      name: "Antoine Sutter",
      email: "antoine.sutter@etu.unige.ch",
    ),
  ),
  doc,
)

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
  det(A) = #a (#e dot #i - #f dot #h) - #b (#d dot #i - #f dot #g) + #c (#d dot #h - #e dot #g) \
  det(A) = (#a dot #(e * i - f * h)) - (#b dot #(d * i - f * g)) + (#c dot #(d * h - e * g)) \
  det(A) = (#(a * (e * i - f * h))) - (#(b * (d * i - f * g))) + (#(c * (d * h - e * g))) \
  det(A) = #(a * (e * i - f * h) - (b * (d * i - f * g)) + (c * (d * h - e * g)))
$

Because the deteminant isn't 0 we know there is a solution. We can now calculate the cofactor matrix of $A$. The cofactor $C i j$ is calculated by eliminating the $i$-th row and $j$-th column, then calculating the determinant of the resulting 2x2 matrix, applying a sign change based on position.

$
  C 11: det(1 dot 0 - 4 dot 6) = -24 \
  C 12: -det(0 dot 0 - 4 dot 5) = 20 \
  C 13: det(0 dot 6 - 1 dot 5) = -5 \
  C 21: -det(2 dot 0 - 3 dot 6) = 18 \
  C 22: det(1 dot 0 - 3 dot 5) = -15 \
  C 23: -det(1 dot 4 - 2 dot 5) = 4 \
  C 31: det(2 dot 4 - 3 dot 1) = 5 \
  C 32: -det(1 dot 4 - 3 dot 0) = -4 \
  C 33: det(1 dot 1 - 2 dot 0) = 1 \
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
  det(A) = (1 dot 5) - (3 dot 2) = 5 - 6 = -1
$

From there we can get $A^(-1)$

$
  A^(-1) = 1 / det(A) dot mat(d, -b; -c, a)
$

With values filled we get the following

$
  A^(-1) = 1 / (-1) dot mat(5, -3; -2, 1)\
  A^(-1) = mat(-5, 3; 2, -1)
$

Now we need to multiply $A^(-1)$ by $B$ to find $X$

$
  X = mat(-5, 3; 2, -1) dot mat(7, 8; 9, 10)\
$

$
  X[0][0] = (-5 dot 7) + (3 dot 9) = -35 + 27 = -8 \
  X[0][1] = (-5 dot 8) + (3 dot 10) = -40 + 30 = -10 \
  X[1][0] = (2 dot 7) + (-1 dot 9) = 14 - 9 = 5 \
  X[1][1] = (2 dot 8) + (-1 dot 10) = 16 - 10 = 6 \
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

$ u #math.dot v = mat(3, 4) #math.dot mat(1, 2) = 3 dot 1 + 4 dot 2 = 11 $

Then doing the dot project $u #math.dot u$ is done as follow:

$ u #math.dot v = mat(3, 4) #math.dot mat(3, 4) = 3 dot 3 + 4 dot 4 = 25 $

After that, we can do the scalar product, which consist of the following:

$ "scalar" = (u #math.dot v) / (u #math.dot u) -> 11 / 25 $

The projection is then a multiplication of the result above with the $u$ vector.

$ w = 11 / 25 dot mat(3, 4) = mat(33 / 25, 44 / 25) #math.approx mat(1.32, 1.76) $

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

```python
r = sum([ui * vi for ui, vi in zip(u, v)])
```

But of course would should use `numpy` for this instead:

```python
r = np.dot(u, v)
```

#pagebreak()

== Orthogonal vector

#let file = read("part_2.py")

Complete the code to construct a vector $v$ orthogonal to the vector $u$ and of the same norm. Comment each line of your code.

#raw(
  file.split("\n").slice(2, 27).join("\n"),
  lang: "python",
  block: true,
)

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

#raw(
  file.split("\n").slice(28, 41).join("\n"),
  lang: "python",
  block: true,
)

In this case, the angle is `0.9746318461970762`.

#pagebreak()

= Eigenvalues, Eigenvectors, and Determinants

== Find the determinant, eigenvectors, and eigenvalues of the matrix

$
  A = mat(
    5, 6, 3;
    -1, 0, 1;
    1, 2, -1;
  )
$

#let a = 5
#let b = 6
#let c = 3
#let d = -1
#let e = 0
#let f = 1
#let g = 1
#let h = 2
#let i = -1

=== Determinant

The deteminant of a 3x3 matrix is defined as such:

$
  mat(
    a, b, c;
    d, e, f;
    g, h, i;
  ) = a e i + b f g + c d h - c e g - b d i - a f h
$

Which gives us the following result

$ #(a * e * i) + #(b * f * g) + #(c * d * h) - #(c * e * g) - #(b * d * i) - #(a * f * h) $
$ 0 + 6 - 6 - 0 - 6 - 10 = -16 $

We can verify using the following python code:

#raw(read("part_3_det.py"), lang: "python", block: true)

And gives us a similar result:

```
-15.999999999999998
```

#pagebreak()

=== Eigen values

#let L = math.lambda

First we need to find the eigen values of $A$. To do so, we need to solve the deteminant for $A - #L I$ where #L is the unknown eigen value and $I$ is the indentity matrix.

$
  A - #L I = mat(
    5, 6, 3;
    -1, 0, 1;
    1, 2, -1;
  ) - #L mat(
    1, 0, 0;
    0, 1, 0;
    0, 0, 1
  ) = 0
$

$
  det mat(
    5-#L, 6, 3;
    -1, -#L, 1;
    1, 2, -1 -#L
  ) = 0
$

We can then compute the deteminant and solve for #L



$
  (5-#L)(-#L)(-1-#L) + 6 - 6 + 3#L +6(-1-#L) - 2(5-#L)\
$

$
  (5-#L)(-#L)(-1-#L) +3#L -6 -6#L -10 +2#L
$

$
  (5-#L)(-#L)(-1-#L) -#L -16
$

$
  (5-#L)(#L+#L^2) -#L -16
$

$
  5#L +5#L^2 -#L^2 -#L^3 -#L -16
$

$
  -#L^3 +4#L^2 +4#L -16
$


Therefore, the polynomial function of $A$ is:

$ p(#L) = -#L^3 +4#L^2 +4#L -16 = 0 $

#let p(x) = (calc.pow(-x, 3) + 4 * calc.pow(x, 2) + 4 * x - 16)

Because this example is being solved by hand, we can safely assume that the roots we're looking for are integers. We also know that the roots have to be factors of the 16 in this example. We can therefore just try out all the factors of 16 and find out if any of them work.

#for i in (1, 2, 4, 8, 16) {
  [
    - $p(#i) = #(p(i))$
    - $p(-#i) = #(p(-i))$
  ]
}

Therefore the eigen values of $A$ are 2, -2 and 4.

#pagebreak()

We can verify that we got it correctly with the following python code

#raw(
  read("part_3_eigvalues.py"),
  lang: "python",
  block: true,
)

```
[-2.  4.  2.]
```

#pagebreak()

=== Eigen vectors

Now that we have the eigen values of $A$, we need to solve the following:

$
  mat(
    5-#L, 6, 3;
    -1, -#L, 1;
    1, 2, -1 -#L
  )
  mat(X;Y;Z) = mat(0;0;0)
$

#let p(x) = $mat(
  #(5 - x), 6, 3;
  -1, #(-x), 1;
  1, 2, #(-1 -x))$

In order to find a solution for $#L = 2$, let's lock $X = 1$.

$ #(p(2)) mat(1;Y;Z) = mat(0;0;0) $

We can get our first 2 equations like so:

$
  f_1 = 3 + 6Y + 3Z = 0\
  f_2 = -1 - 2Y +Z = 0
$

Then $f_1 - 3f_2$ to extract $Y$

$
  f_1 - 3f_2 = (3 + 6Y + 3Z) - (-3 - 6Y + 3Z) = 6 + 12Y = 0\
  12Y = -6\
  Y = -6 / 12 = -1 / 2
$

Now let's insert $Y$ into $f_2$:

$
  f_2 = -1 -2(-1 / 2) + Z = 0\
  f_2 = -1 + 1 + Z = 0\
  Z = 0
$

We can now verify with $f_3$:

$
  1 + 2(-1 / 2) = 0\
  1 - 1 = 0
$

Therefore, our eigen vector for the eigen value $#L = 2$ is:

$
  mat(1; -1/2; 0)
$

We can scale it up to get rid of the fraction

$
  mat(2; -1; 0)
$

We can now verify that our original equation is true:

$ #(p(2)) mat(2; -1; 0) = mat(0;0;0) $

#pagebreak()

We can then verify we got it correctly using the following python code:

#raw(
  read("part_3_eigvec.py"),
  lang: "python",
  block: true,
)

```
[[ 3.  6.  3.]
 [-1. -2.  1.]
 [ 1.  2. -3.]]
[ 1.  -0.5  0. ]
[0. 0. 0.]
```

#pagebreak()

Now we need to do the same for $#L = -2$. This time we lock $Y = 1$ because locking $X = 1$ would lead too all variables disapearing when doing subscractions, meaning $X$ is 0 which we will confirm is true later. I will not comment every steps from now one since it's similar to the previous example.

$ #L = -2 -> #(p(-2)) mat(X;1;Z) = mat(0;0;0)\ $
$ f_1 = 7X + 6 + 3Z = 0 $
$ f_2 = -1X + 2 + Z = 0 $

$
  f_1 - 3f_2 = (7X + 6 + 3Z) - (-3X + 6 + 3Z) = 10X = 0\
  X = 0
$

$
  f_2 = 2 + Z = 0\
  Z = -2
$

$
  f_3 = 2 - 2 = 0
$

Therefore the eigen vector for $#L = -2$ is:

$ mat(0; 1; -2) $

And we can again confirm via python that the result is correct.

#raw(
  read("part_3_eigvec_2.py"),
  lang: "python",
  block: true,
)

```
[[ 7.  6.  3.]
 [-1.  2.  1.]
 [ 1.  2.  1.]]
[ 0  1 -2]
[0. 0. 0.]
```

#pagebreak()

Now we need to do the same for $#L = 4$

$ #L = 4 -> #(p(4)) mat(1;Y;Z) = mat(0;0;0) $

$ f_1 = 1 + 6Y + 3Z = 0 $
$ f_2 = -1 -4Y + Z = 0 $
$ f_1 - 3f_2 = (1 + 6Y + 3Z) - (-3 - 12Y + 3Z) = 18Y + 4 = 0 $
$ Y = -4 / 18 = -2 / 9 $
$ f_2 = -1 - 4(-2 / 9) + Z $
$ f_2 = -1 - 4(-2 / 9) + Z $
$ f_2 = -1 + 8 / 9 + Z = 0 $
$ Z = 1 - 8 / 9 $
$ f_3 = 1 + 2(-2 / 9) - 5(1-8 / 9) = 0 $

Therefore the eigen vector for $#L = 4$ is:

$ mat(1; -2/9; 1-8/9) $

#pagebreak()

And we can again confirm via python that the result is correct. This time however we do not get exactly the vector `[0. 0. 0]` but this is because of the limitation of floating point numbers. We are close enough.

#raw(
  read("part_3_eigvec_3.py"),
  lang: "python",
  block: true,
)

```
[[ 1.  6.  3.]
 [-1. -4.  1.]
 [ 1.  2. -5.]]
[ 1.         -0.22222222  0.11111111]
[ 2.22044605e-16  0.00000000e+00 -2.22044605e-16]
```

#pagebreak()

== Positive semidefinite proof

The covariance matrix for the $n$ samples $x_1,...,x_n$, each represented by a $d #math.times 1$ column vector, is given by

$
  C = 1 / (n-1) sum_(i=1)^n (x_i - #math.mu)(x_i - #math.mu)^T
$

where $C$ is a $d #math.times d$ matrix and $#math.mu = sum_(i=1)^n x_i/n$ is the sample mean. Prove that $C$ is always positive semidefinite. (Note: A symmetric matrix $C$ of size $d #math.times d$ is _positive semidefinite_ if $v^T C v >= 0$ for every $d #math.times 1$ vector $v$.)

=== $C$ is symmetric

We can see that $C$ is symmetric because $(x_i - #math.mu)(x_i - #math.mu)^T$ is a $d #math.times d$ matrix and symmetric itself because transposing it gives back the same result.

$
  ((x_i - #math.mu)(x_i - #math.mu)^T)^T = (x_i - #math.mu)(x_i - #math.mu)^T
$

Therefore, $C$ being a sum of symmetric matrices, $C$ is symmetric.

=== Positive semidefinite

Proving that $C$ is semidefinite means that $v^T C v >= 0$ where $v$ is a $d #math.times 1$ vector. We can then do the following:

$
  v^T C v = v^T mat(1/(n-1) sum_(i=1)^n (x_i - #math.mu)(x_i - #math.mu)^T ) v
$

We can move the multiplication and sum outside:

$
  v^T C v = 1 / (n-1) v^T mat(sum_(i=1)^n (x_i - #math.mu)(x_i - #math.mu)^T ) v
$

From there, move $v^T$ inside the sum:

$
  v^T C v = 1 / (n-1) sum_(i=1)^n v^T (x_i - #math.mu)(x_i - #math.mu)^T v
$

We can then factor $v$ and simplify like this:

$
  v^T C v = 1 / (n-1) sum_(i=1)^n (v(x_i - #math.mu)^T)^2
$

Because squaring can only produce positive values, we know that $C$ is semipositive.

$
  v^T C v = 1 / (n-1) sum_(i=1)^n (v(x_i - #math.mu)^T)^2 >= 0
$

#pagebreak()

== Eigenvalues of covariance matrices

#figure(
  image("part_3_3.png"),
  caption: [Eigenspecta of the data sets],
)

The determinant of a covariance matrix is equal to the product of its eigenvalues. Large eigenvalues indicate directions with high variance, while small or zero eigenvalues suggest low or no variance, indicating potential linear dependence. The spectrum of eigenvalues helps understand the dimensionality and variance structure: a steep drop-off implies effective lower-dimensional representation, while a gradual decline indicates spread across dimensions. Differences in eigenvalues between datasets arise from intrinsic dimensionality, data variance, feature correlations, and noise levels, providing insights into each dataset's properties.

#pagebreak()

== Signular matrix for $lambda = 0$

An eigenvalue $lambda$ of a matrix $A$ is defined as a scalar such that there exists a non-zero vector $x$ for which the following equation is true:

$ A x = lambda x $

Imagine the case $lambda = 0$:

$
  A x = 0 x\
  A x = 0
$

This implies that the matrix $A$ has a non-zero vector $x$ in its null space. In other words, there is some non-zero vector $x$ such that $A x = 0$.
By definition, a matrix $A$ is singular if it does not have full rank, or equivalently, if its null space contains a non-zero vector. Since we have found a non-zero vector $x$ such that $A x = 0$, this implies that $A$ is singular.

== Proof that the determinant of $A$ must be zero if any eigenvalue is zero

The determinant of a matrix $A$, denoted as $ det(A)$, can be expressed in terms of its eigenvalues. For an $n times n$ matrix, if $lambda_1, lambda_2, ..., lambda_n$ are the eigenvalues of $A$, then the determinant of $A$ is given by the product of its eigenvalues:

$
  det(A) = lambda_1 lambda_2 ... lambda_n
$

Therefore, if any of the eigen value is 0, the product will be 0 and the deteminant will also be 0.

#pagebreak()

= Computing Projection Onto a Line

There are given a line $alpha : 3x + 4y = -6$ and a point $A$ with the coordinates $(-1, 3)$.

== Distance from point $A$

Let's first rewrite $alpha$ as such:

$alpha: 3x + 4y + 6 = 0$

Then we can use the Point-Line Distance Formula, defined as such:

$ d = (|a x_1 + b y_1 + c|) / sqrt(a^2 + b^2) $

Given the following variables:

$
  a = 3, b = 4, c = 6\
  x_1 = -1, y_1 = 3
$

We get the following equation that we can then simpliy to get our result.

$
  d = (3 dot -1 + 4 dot 3 + 6) / (sqrt(3^2 + 4^2))\
  d = 3(-1) + 4(3) + 6\
  d = -3 + 12 + 6\
  d = 15
$

Therefore the distance from $A$ to $alpha$ is 3.

#pagebreak()

== Python visualized

We can visualize the projection of $A$ using the following python code:

#raw(
  read("part_4.py"),
  lang: "python",
  block: true,
)

#figure(
  image("part_4.png"),
  caption: [Projection of point A],
) <plot>

asdsd
