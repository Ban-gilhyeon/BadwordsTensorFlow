package example.com.badwordsspringproject.service.components;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class OnnxInferenceService {
    private  OrtEnvironment env;
    private  OrtSession session;

    @Value("${onnx.model.path}")
    private String modelPath;

    @PostConstruct
    public void init() throws OrtException {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        session = env.createSession(modelPath, options);
    }

    //int64 입력을 LongBuffer로 받아 ONNX 추론 수행
    public float[][] runInference(long[] inputIdsData, long[] attentionMaskData, long[] inputShape) throws OrtException {
        // long 배열을 LongBuffer로 감싸기
        LongBuffer inputIdsBuffer = LongBuffer.wrap(inputIdsData);
        LongBuffer attentionMaskBuffer = LongBuffer.wrap(attentionMaskData);

        // OnnxTensor를 생성할 때, OnnxJavaType.INT64를 지정하여 int64 텐서를 생성합니다.
        OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIdsBuffer, inputShape, OrtJavaType.INT64);
        OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, attentionMaskBuffer, inputShape, OrtJavaType.INT64);


        // 모델의 입력 이름은 ONNX 변환 시 지정한 이름과 정확히 일치해야 합니다.
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", inputIdsTensor);
        inputs.put("attention_mask", attentionMaskTensor);

        OrtSession.Result results = session.run(inputs);
        float[][] logits = (float[][]) results.get(0).getValue();

        // 사용한 텐서 자원 해제
        inputIdsTensor.close();
        attentionMaskTensor.close();
        return logits;
    }

    public void close() throws OrtException {
        session.close();
        env.close();
    }
}
