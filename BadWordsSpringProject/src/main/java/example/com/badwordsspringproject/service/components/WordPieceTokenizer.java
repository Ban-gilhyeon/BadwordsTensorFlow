package example.com.badwordsspringproject.service.components;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.util.JsonUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.huggingface.HuggingfaceChatModel;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@Component
@Slf4j
public class WordPieceTokenizer {
    HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("beomi/kcbert-base");

    String text = "느림보 거북이";

    Encoding encoding = tokenizer.encode(text);

    long[] tokenIds = encoding.getIds();
    String[] tokens = encoding.getTokens();

}
