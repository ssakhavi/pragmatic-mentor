---
tags: ["ML Architect", "Experience"]
date: "2020-11-25T00:00:00Z"
description: How breaking the code on purpose can lead to learning where in the code
  there are problems.
title: Breaking the Code to Learn
toc: false
---


I've started a new job at [AI Singapore](https://www.aisingapore.org). The nature of the job is in a way that I have to mentor people and guide them to be better AI Engineers.

That's why I think my blog posts will be more relevant and useful.

Today, I want to touch upon a topic that came to my mind when I was thinking about how to teach people whether there ML system is robust enough or not.

When I was in my PhD, I would browse around to learn new stuff and one of the most amazing website I came across to learn new programming languages was "Learn X the Hard Way". At that time *X* was **C** for me: CUDA had just gained popularity and because CUDA is mainly written in C and was keen in understanding parallel programming on GPUs, I decided to learn some C via the mentioned website. The website has changed since then ([link](https://learncodethehardway.org/)) and most of the material which were free need to be purchased now.

The way that the author of the learning course would teach the language was ineteresting: First, he would introduce a concept and explain how that concept is used for the C language. But later, he would ask to modify different places of the code and make the code *break*. This breaking of the code would result in an error message. By asking to break the code, the learners would be able to identify  error messages corresponding to different problems that would be raised in different situations.

Now, When I was thinking about robustness in machine learning and how to make sure the code is written in a way that it doesn't crash, I came up with this solution in my mind: What if we modify some parts of the configuration or the data to see whether the pipeline is working or not and whether the different parts of the code have good handling of unexpected inputs of configurations.

You could say that this sounds a lot like testing. I would say: yes! usually for test we check the functionality and whether for a certain input, the output is okay. Eventually, breaking the code will lead to writing tests and when you become more experienced, you know what tests you need to write for your code.

Also, you might want to look at this as an attack on the network to check whether the code has covered the different scenarios of the configuration and data input.

Let's talk about an example: One of the mistakes that I have made many times, during my PhD and my KK days, is the matter of reading data based on a certain pattern ( i.e. asusming the the files are in sequence). This isn't true in most of the times. One way to break the code in this case is to remove a data point manually and see if the data pipeline still works.

Actually, I've realized that pytorch-lightning does some kind of validation before it starts training the data to see whether the config, input and the model match.

Takeaway:

* Sometimes you need to break the code to understand the problem and design your code in a better way.
* Don't write your code and assume because it's working, you don't have any bugs. Bugs won't show if you don't give your code a bit of a shake.
