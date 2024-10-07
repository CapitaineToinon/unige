#import "@local/unige:0.1.0": conf

#show: doc => conf(
  title: "The Quadratic Assignment Problem",
  header: "Université de Genève",
  subheader: [
    Metaheuristics for Optimization\
    14X013
  ],
  authors: (
    (
      name: "Antoine Sutter",
      email: "antoine.sutter@etu.unige.ch",
    ),
  ),
  doc,
)

#let code = read("main.py")

#let snippet(start, end) = {
  let lines = code.split("\n")

  raw(
    lines.slice(start - 1, end).join("\n"),
    lang: "python",
    block: true,
  )
}

= Code explaination

This report won't cover the code line by line as the code already has comments and is pretty safe explainatory. The code is available in the `main.py` file that should be next to this report. It is also available at the end of this report @code.

= Results

The algorythm has been implemented to accept a variable tenure as well as the option to enable diversification or not. In order to judge this impact of both of these, the algorythm has been ran 10 times for each tenure in ${1, 0.5n, 0.9n}$ where $n$ is the width of the square matrix, each time with and without diversification. The algorythm internally iterates 500 times before stopping and returning whatever the best solution is at the time.

#figure(
  image("out.png"),
  caption: [Result per tenure,\
    with and without diversification],
) <results>

#pagebreak()

As we can see in @results, when the tenure is low, the results aren't great and diversification doesn't seem to be able to help much. However with increased tenure, the results impove and diversification starts having an impact on the results. However the results vary greatly from the initial element which in this case is a random permuation.

== Improvements

Some things could be improved in this code. As highlighted in the code's comment, we currently only return a single solution. However, it is possible to find 2 solutions with the same fitness result and we could return a list of solutions instead.

#snippet(165, 170)

We could also think about implementing some caching to avoid duplicated calculations for the code to perform better. This would allows us to increase the amount of internal iterations for better results.

#pagebreak()

= Questions

== In this specific case, what is the neighborhood of an element of the research space?

An element is defined a permutation of size $n$, meaning all unique numbers from $0..n$ in a list. The neighborhood of an element is defined as all the possible unique swaps of 2 values in that list. In this project for example here is how the function that renerates all the swaps is defined:

#snippet(46, 54)

So for $n=5$, you would get the following output:

```
[
  (0, 1), (0, 2), (0, 3), (0, 4),
  (1, 2), (1, 3), (1, 4),
  (2, 3), (2, 4),
  (3, 4)
]
```

== In terms of $n$ (number of locations/facilities), what is the size of the neighborhood? (i.e., how many neighbors does each permuation have?)

The size of the neighborhood is $n$ chose 2. Given the code written above, it can be defined as

$
  S = n(n-1) / 2
$

#pagebreak()

= Main.py <code>

#raw(
  code,
  lang: "python",
  block: true,
)