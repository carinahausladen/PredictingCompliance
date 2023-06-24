from ._builtin import Page, WaitPage
from .models import Constants
from .pages_abstrct import PageStage1, PageStage2


class intro_gen(Page):
    timer_text = 'Bitte treffen Sie eine Auswahl innerhalb von:'
    timeout_seconds = 5 * 60

    def before_next_page(self):

        self.group.absent_group = False
        self.player.absent_player = False  # set initial values to avoid errors when testing
        for p in self.group.get_players():
            p.control_correct = True

        if self.timeout_happened:#
            self.group.absent_group = True
            self.player.absent_player = True


class iban(PageStage1):
    form_model = 'player'
    form_fields = ['forename', 'surname', 'iban', 'bic']


class intro_spec(PageStage1):
    pass


class check_understanding_0(PageStage1):
    timeout_seconds = 7 * 60

    form_model = 'player'
    form_fields = ['contr_partner_0', 'contr_prob_0',
                   'contr_fix_0', 'contr_var_0', 'contr_fine_0',
                   'contr_50noaudit_0', 'contr_50audit_0',
                   'contr_65noaudit_0',
                   'contr_35audit_0']

    def before_next_page(self):
        super().before_next_page()
        self.player.control_correct = self.player.contr_partner_0 == 2 and \
                                      self.player.contr_prob_0 == 1 and \
                                      self.player.contr_fix_0 == 1 and \
                                      self.player.contr_var_0 == 1 and \
                                      self.player.contr_fine_0 == 2 and \
                                      self.player.contr_50noaudit_0 == 60 and \
                                      self.player.contr_50audit_0 == 60 and \
                                      self.player.contr_65noaudit_0 == 70 and \
                                      self.player.contr_35audit_0 == 50


class check_understanding(Page):
    timer_text = 'Bitte treffen Sie eine Auswahl innerhalb von:'
    timeout_seconds = 4 * 60

    def is_displayed(self):
        return self.player.control_correct == False and self.group.absent_group == False

    form_model = 'player'
    form_fields = ['contr_partner', 'contr_prob',
                   'contr_fix', 'contr_var', 'contr_fine',
                   'contr_50noaudit', 'contr_50audit',
                   'contr_65noaudit',
                   'contr_35audit']


class surplus_hours(PageStage1):
    pass


class chat_wait_before(WaitPage):

    def is_displayed(self):
        return self.group.absent_group == False

    body_text = "Bitte warten Sie, bis der anderen Mitarbeiter Ihrer Zweiergruppe die Verständnisfragen beantwortet hat. " \
                "Im Anschluss können Sie beide chatten, " \
                "um sich über die Eingabe der Überstunden auszutauschen."


class chat(Page):
    timer_text = 'Bitte beenden Sie den Chat in:'
    timeout_seconds = 7 * 60

    def is_displayed(self):
        return self.group.absent_group == False

class state_hours(PageStage1):
    form_model = 'player'
    form_fields = ['hours_stated']


class calc(WaitPage):
    def is_displayed(self):
        return self.group.absent_group == False

    after_all_players_arrive = 'hours_class_payoff'


class questions_prep(PageStage2):
    pass


class questions_belief(PageStage2):
    form_model = 'player'
    form_fields = ['belief']


class questions_socio(PageStage2):
    form_model = 'player'
    form_fields = ['age', 'gender', 'study', 'study_level', 'econ']


class questions_moral(PageStage2):
    form_model = 'player'
    form_fields = ['moral_freude', 'moral_aerger', 'moral_angst', 'moral_scham', 'moral_schuld']


class questions_other(PageStage2):
    form_model = 'player'
    form_fields = ['risk', 'lie', 'complex', 'pol', 'income', 'pray', 'lab_exp']


class questions_expdem(PageStage2):
    form_model = 'player'
    form_fields = ['exp_demand_q', 'exp_demand_open', 'exp_demand_chat', 'exp_demand_chat_open']


class payoff_show(Page):
    def is_displayed(self):
        return self.group.absent_group == False and self.player.absent_player == False


class outro(Page):
    def is_displayed(self):
        return self.group.absent_group == False and self.player.absent_player == False


class calc_alternative(Page):
    timeout_seconds = 3

    def is_displayed(self):
        return self.group.absent_group == True or self.player.absent_player == True

    def before_next_page(self):
        if self.player.absent_player:
            self.player.payoff = Constants.showup
        else:
            self.player.payoff = Constants.showup + Constants.income_fix / 10


class outro_alternative(Page):

    def is_displayed(self):
        return self.group.absent_group == True or self.player.absent_player == True


page_sequence = [
    intro_gen,
 #   iban,
    intro_spec,

    check_understanding_0,
    check_understanding,

    surplus_hours,

    chat_wait_before,
    chat,

    state_hours,
    calc,

    questions_prep,
    questions_belief,
    questions_socio,
    questions_moral,
    questions_other,
    questions_expdem,

    payoff_show,

    outro,

    calc_alternative,
    outro_alternative
]
