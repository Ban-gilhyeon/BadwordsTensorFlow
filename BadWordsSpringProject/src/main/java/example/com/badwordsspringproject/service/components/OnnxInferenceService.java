package example.com.badwordsspringproject.service.components;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class OnnxInferenceService {
    private final OrtEnvironment env;
    private final OrtSession session;

    //int64 입력을 LongBuffer로 받아 ONNX 추론 수행
    public float[][] runInference(
            long[] inputIdsData,
            long[] attentionMaskData,
            long[] inputShape) throws OrtException {
        LongBuffer inputIdsBuffer = LongBuffer.wrap(inputIdsData);
        LongBuffer attentionMaskBuffer = LongBuffer.wrap(attentionMaskData);

        OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIdsBuffer, inputShape);
        OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, attentionMaskBuffer, inputShape);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("inputIds", inputIdsTensor);
        inputs.put("attentionMask", attentionMaskTensor);

        OrtSession.Result results = session.run(inputs);
        float[][] logits = (float[][]) results.get(0).getValue();

        inputIdsTensor.close();
        attentionMaskTensor.close();
        return logits;
    }

    public void close() throws OrtException {
        session.close();
        env.close();
    }
}
