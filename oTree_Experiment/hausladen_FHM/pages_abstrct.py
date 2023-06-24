from ._builtin import Page


class PageStage1(Page):

    timer_text = 'Bitte treffen Sie eine Auswahl innerhalb von:'
    timeout_seconds = 5 * 60

    def is_displayed(self):
        return self.group.absent_group == False

    def before_next_page(self):
        if self.timeout_happened:
            self.group.absent_group = True
            self.player.absent_player = True


class PageStage2(Page):
    timeout_seconds = 5 * 60

    def is_displayed(self):
        return self.group.absent_group == False and self.player.absent_player == False  # I changed the or to and

    def before_next_page(self):
        if self.timeout_happened:
            self.player.absent_player = True
