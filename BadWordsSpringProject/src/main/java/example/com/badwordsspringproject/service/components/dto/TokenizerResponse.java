package example.com.badwordsspringproject.service.components.dto;

public record TokenizerResponse(
        long[] tokenIds,
        String[] tokens,
        long[] attentionMask
) {
    public static TokenizerResponse fromTokens(long[] tokenIds,String[] tokens,long[] attentionMask) {
        return new TokenizerResponse(tokenIds, tokens, attentionMask);
    }
}
