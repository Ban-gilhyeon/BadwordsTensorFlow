package example.com.badwordsspringproject.service;

import ai.onnxruntime.OrtException;
import example.com.badwordsspringproject.service.components.OnnxInferenceService;
import example.com.badwordsspringproject.service.components.TokenizerService;
import example.com.badwordsspringproject.service.components.dto.TokenizerResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class BadWordFilterService {
    private final OnnxInferenceService onnxInferenceService;
    private final TokenizerService tokenizerService;

    private static final float THRESHOLD = 0.75f;

    public boolean isProfane(String plainText) {
        try {
            TokenizerResponse response = tokenizerService.getTokenizer(plainText);
            if(response == null || response.tokenIds() == null) {
                return true;
            }

            long[] tokenIds = response.tokenIds();
            long[] attentionMask = response.attentionMask();
            long[] inputShape = new long[]{1, tokenIds.length};

            float[][] logits = onnxInferenceService.runInference(tokenIds, attentionMask, inputShape);

            float probability = (float) (1 + Math.exp(-logits[0][0]));

            return probability >= THRESHOLD;
        }catch (OrtException e) {
            e.printStackTrace();
            return true;
        }
    }
}
