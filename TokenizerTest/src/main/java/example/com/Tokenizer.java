package example.com;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.nio.file.Paths;

public class Tokenizer {

    public static RequestToken getTokenizer(String text) {
        try {
            HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(
                    Paths.get("/Users/bangilhyeon/git/BadwordsTensorFlow/saved_model/tokenizer.json")
            );

            Encoding encoding = tokenizer.encode(text);
            return RequestToken.fromTokens(encoding.getIds(), encoding.getTokens());
        }catch (Exception e){
            e.printStackTrace();
        }
        return null;
    }
}
