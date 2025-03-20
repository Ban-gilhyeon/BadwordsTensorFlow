package example.com.badwordsspringproject.api;

import example.com.badwordsspringproject.service.BadWordFilterService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class BadWordFilterController {
    private final BadWordFilterService badWordFilterService;

    @GetMapping("/check")
    public Boolean checkBadWord(@RequestParam("word") String word) {
        return badWordFilterService.isProfane(word);
    }
}
