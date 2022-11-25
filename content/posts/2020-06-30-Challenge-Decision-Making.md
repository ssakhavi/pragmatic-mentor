---
tags: ["ML Architect","Tools", "Decisions"]
date: "2020-06-30T00:00:00Z"
description: The challenge of making decisions with the existence of different tools
title: 'Challenge: Decision Making'
author: Siavash Sakhavi
comments: true
---

In this blog I'm going to discuss one of the most difficult challenge that any architect will come across in their path to becoming professional: deciding on design and toold choices. For myself, this challenge reoccurs more and more often becuase of my own tendancy to making sure that my design is right and that I'm using the right tools. Maybe it's better that I delve into what I mean with an example.

## Choosing a high level PyTorch tool

It was around a two months ago when I was given new data to work with and I was tasked to train a segmentation model on the data. The task was simple: we have class A which is our desired class and everything else is background. This seems very simple and straightforward until a person like me starts thinking about how I can make the whole thing in the best way possible.

I had done this experiment before a year back and since then, I knew there were new tools that could be used to bring more "structure" into my code. So I started searching.

After some searching I came across a few frameweorks which are built on top of PyTorch and are designed to make life easier for those who don't want to have a lot of boilerplate code. You can find almost all of them in the [PyTorch EcoSystem page](https://pytorch.org/ecosystem/). After finding these amazingly well-designed frameworks, I had to make the tough decision: which one should I use.

## Narrowing it down

Here it got a bit tricky: I was trying to find something to use which was easy to plug into my current code and at the same time, had structure. My philosophy is that a good framework is a framework which enforces structure. That's why after looking at the examples and source code, I narrowed down my options to two: **PyTorch Lightning (PL)** and **Catalyst**.

Initially, I had a tough time; Both frameworks are amazing and they cover a lot of the boilerplate code and give some degrees of autmation. Catalysy even has a example on how to use their framework for segmentation. But in order to come up with a decision, I decided to to implement the segmentation algorithm in both frameworks.

One of the things that was intriguing about PL and eventually led to me deciding on this framework, was the way it was implemented: The whole framework was a python class which needed its abstract methods to be populated based on the problem in hand. This was exactly what I was looking for. After populating the class (PL module), a simple method was executed with the intended parameters and logging methods. This really helps with making the training modular as possible, which is one of the needs for a flexible architecture. I saw this benefit when I tried mixing [hydra](https://hydra.cc/) with my PL code. It was supper easy because the parameter values were already being passed as a dicitonary and I just needed to pass them as `DictConfig` (the type of the Hydra parameter dictionary based on OmegaConf).

Then I added MLFlow into the mix. This was another reason I selected PL: It had a built-in MLFlow logger. By adding MLFlow, I found a [bug](https://github.com/PyTorchLightning/pytorch-lightning/issues/2058) which most likely was spotted because of the use of Hydra and MLFlow together. This resulted in a [PR](https://github.com/PyTorchLightning/pytorch-lightning/pull/2216). I really don't know if it was a bug or not because it was possible that upgrading the versions of my python packages would solve the problem, but they updated the code in a way that solved it entirely.

## Takeaways

My Takeaways from this experience:

- Believe in a philosophy or an ideal regarding how code should look like.
- Narrow down options by intially reading through the documents and sample code
- After narrowing down, impelement the raw version of your code in the options you have and see the benefits and caveats
- Don't be shy to submit issues or PRs on GitHub (I submitted a PR, panicked and deleted my code and one of the owners asked that why I did that)

Take care.
