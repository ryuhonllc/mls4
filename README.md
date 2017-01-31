# TBDSG Machine Learning Session 4

In this session we will finally get to TensorFlow, but we're not actually
doing anything with neural networks this time around. We've looked at the history
of neural networks (session 1) and the support tools in python (session 2).
We delved into the components of a pre-deep learning era neural network last 
time (in session 3).

Deep networks are complex, and when they don't work, it helps to know how
the platform they use works for debugging purposes. Thus, before we
get into deep learning, we're going to look into a supporting platform.
For simplicity, we'll look at TensorFlow, although there are many other choices.

## Benefits of TensorFlow
* it has name recognition
* lots of effort has gone into explaining it
* lots of debugging has gone into it
* it's under active development by google

## Disadvantages of TensorFlow
* It's big! And therefore can be hard to wrap one's head around.
* Installation can be a pain, particularly with GPU support
* It's API doesn't line up with others such as numpy.
* It's API is inconsistent (e.g. case in `tf.Variable` vs `tf.constant`)

One must start somewhere, though. In this session, we will focus on
non-gpu installation.  However, if you have a system with a GPU, feel
free to bring it in. We'll try to have a gpu system available
to do a speed comparison.

# Part A: Installation

There are lots of guides out there, so this section will focus on how to set things
up to work with the unit tests we'll be using in Part B.

## environment setup

Clone this repo, of course. Then,

```
  mkvirtualenv -p `which python3` mls4
  pip install -r requirements.txt
```

## download and install

Go [Here](https://www.tensorflow.org/get_started/os_setup#virtualenv_installation) and
select the version appropriate for your system. We'll be working with python 3.5, although
the tutorial is not particularly 3.5 specific.



# Part B: Exploring TensorFlow via unit tests


Rather than lecturing on TF, I'm providing a file with some unit tests.
We'll go through these tests and use them to drive the discussion.

