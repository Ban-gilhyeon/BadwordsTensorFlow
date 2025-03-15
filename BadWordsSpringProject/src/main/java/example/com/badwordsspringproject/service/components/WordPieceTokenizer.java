package example.com.badwordsspringproject.service.components;

import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class WordPieceTokenizer {
    private Map<String, Integer> vocab;
    private String unkToken = "[UNK]";
    private int maxInputCharsPerWord = 100;


}
