# ising-compiler [![Build Status](https://travis-ci.com/fancompute/ising-compiler.svg?branch=master)](https://travis-ci.com/fancompute/ising-compiler)

🍰 Compiling your code to an Ising Hamiltonian so you don't have to!

![Computing 1+1=2 in a fantastically roundabout manner](https://raw.githubusercontent.com/fancompute/ising-compiler/master/assets/oneplusone.gif)

## About

This library was a final project for Stanford's graduate statistical mechanics class. The Ising model, despite its simplicity, has a rich array of properties that allow for universal computation. Each elementary Boolean logic gate can be implemented as an Ising system of 2-4 spins. This library allows you to compile a sequence of Boolean logic gates into a spin system where the result of the computation is encoded in the ground state of the Hamiltonian. I provide several demonstrations of compiling complex circuits into Ising spin systems and use Monte Carlo simulations to show that the compiled circuits encode the desired computations. 

See the [paper](https://github.com/fancompute/ising-compiler/blob/master/paper.pdf) for more details.

## Examples

See this [example notebook](https://github.com/fancompute/ising-compiler/blob/master/examples.ipynb) for demonstrations of how this library can be used.