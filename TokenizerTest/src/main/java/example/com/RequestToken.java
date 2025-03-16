package example.com;

public record RequestToken(
        long[] tokenIds,
        String[] tokens
) {
    public static RequestToken fromTokens(long[] tokenIds,String[] tokens) {
        return new RequestToken(tokenIds, tokens);
    }
}
