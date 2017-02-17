This is yet another implemntation of A Neural Algorithm of Artistic Style (https://arxiv.org/abs/1508.06576).

Fast methods like texture nets and other will be implimented later 

Intially it bases on https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/   [1]

Key features are following:

1. Instead vgg based content loss  Frabnius norm of difference beween images  was used.

2. Optimization by adam gradient decent.

3. Special constraint on rgb colors (gray scale only option)

![Alt text](https://pp.vk.me/c637326/v637326831/2f93c/mjyPwpe-iDY.jpg?raw=true "Example")
