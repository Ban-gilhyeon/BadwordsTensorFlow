package example.com.badwordsspringproject.service.components;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import example.com.badwordsspringproject.service.components.dto.TokenizerResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.nio.file.Paths;

@Component
@Slf4j
public class TokenizerService {
    public static TokenizerResponse getTokenizer(String text){
        try{
            HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(
                    Paths.get("/Users/bangilhyeon/git/BadwordsTensorFlow/saved_model/tokenizer.json")
            );

            Encoding encoding = tokenizer.encode(text);
            // 여기서 attention mask를 생성: tokenIds가 0이면 0, 아니면 1
            long[] tokenIds = encoding.getIds();
            long[] attentionMask = createAttentionMask(tokenIds);

            return TokenizerResponse.fromTokens(tokenIds, encoding.getTokens(), attentionMask);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private static long[] createAttentionMask(long[] tokenIds){
        long[] mask = new long[tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++){
            mask[i] = tokenIds[i] == 0 ? 0 : 1;
        }
        return mask;
    }
}
