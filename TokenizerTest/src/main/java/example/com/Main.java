package example.com;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.lang.reflect.Array;
import java.nio.file.Paths;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        String text = "느림보거북";

        RequestToken token = Tokenizer.getTokenizer(text);
        System.out.println("tokenIds : " + java.util.Arrays.toString(token.tokenIds()));
        System.out.println("tokens : " + java.util.Arrays.toString(token.tokens()));
    }
}