package com.gabrielfreire.app;

/**
 * Created by Gabriel Freire on 10/04/2017.
 */
public class ActivationFunction {

    public static float sigmoid(float x){
        return (float) (1 / (1+Math.exp(-x)));
    }
    //derivative sigmoid
    //no need to call the sigmoid method again since we will only call the derivative after calling sigmoid function
    public static float dSigmoid(float x){
        return x * (1-x);
    }
}
