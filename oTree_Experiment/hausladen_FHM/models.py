import random

from jinja2 import Environment, FileSystemLoader, select_autoescape
from otree.api import (
    models,
    widgets,
    BaseConstants,
    BaseSubsession,
    BaseGroup,
    BasePlayer
)
from schwifty import IBAN, BIC

author = 'Carina Ines Hausladen'

doc = """
Participants state surplus-hours in groups of 2.
1/3 of groups is controlled.
If the amount stated is not the same, the group is always controlled.
If controlled and surplus hours were not truthfully reported, the participant pays a fine.
"""


class Constants(BaseConstants):
    name_in_url = 'FHM'
    players_per_group = 2
    num_rounds = 1
    income_fix = 40
    hours_max = 60
    actual_surplus_hours = 10
    showup = 2
    control_correct_euro = 1
    soft_timeout = 4 * 60 * 1000  # 4 min
    hard_timeout = 5 * 60  # 5 min
    instructions_template = 'hausladen_FHM/intro_spec_template.html'


class Subsession(BaseSubsession):
    def creating_session(self):
        groups_all = []

        for group in self.get_groups():
            groups_all.append(group)

        groups_to_audit = random.choices(groups_all, k=int(round(len(groups_all)*0.3, 0)))

        for group in self.get_groups():

            group.all_groups = len(groups_all)  # set these variables to call in int_spec
            group.audit_groups = len(groups_to_audit)

            if group in groups_to_audit:
                group.to_audit = True
            else:
                group.to_audit = False

        def generate_instructions_html():
            env = Environment(loader=FileSystemLoader("hausladen_FHM/templates/hausladen_FHM"),
                              autoescape=select_autoescape(["html", "xml"])
                              )

            template = env.get_template("intro_spec_template.html")
            rendered_html = template.render({"Constants": Constants,
                                             "group": self.get_groups()[0]})
            with open("_static/intro_spec_static.html", "w") as f:
                f.write(rendered_html)

        generate_instructions_html()


class Group(BaseGroup):
    all_groups = models.IntegerField()
    audit_groups = models.IntegerField()

    to_audit = models.BooleanField()
    absent_group = models.BooleanField()

    def hours_class_payoff(self):
        hours_both = []
        for p in self.get_players():
            hours_both.append(p.hours_stated)

        for p in self.get_players():
            if hours_both[0] == hours_both[1]:
                p.same_hours = True
            else:
                p.same_hours = False

            p.to_audit_player = self.to_audit

            p.cal_payoff()


class Player(BasePlayer):
    absent_player = models.BooleanField()
    control_correct = models.BooleanField()
    same_hours = models.BooleanField()
    to_audit_player = models.BooleanField()
    audited_both = models.BooleanField()

    forename = models.StringField(label="Vorname")
    surname = models.StringField(label="Nachname")
    email = models.StringField(label="Ihre Email-Adresse")
    iban = models.StringField(label="IBAN")

    def iban_error_message(self, value):
        try:
            _ = IBAN(value)
        except ValueError as e:
            return f'Nicht korrekt. Bitte versuchen Sie es erneut. ({str(e)})'

    bic = models.StringField(label="BIC")

    def bic_error_message(self, value):
        try:
            _ = BIC(value)
        except ValueError as e:
            return f'Nicht korrekt. Bitte versuchen Sie es erneut. ({str(e)})'

    hours_stated = models.IntegerField(
        min=0,
        max=Constants.hours_max,
        label="Ich habe folgende Anzahl an Überstunden geleistet:"
    )

    # Control questions_0
    contr_partner_0 = models.IntegerField(
        choices=[
            [1, 'Nichts'],
            [2, 'Ihre Zweiergruppe wird definitiv überprüft']
        ],
        label="Was passiert, wenn Sie etwas anderes angeben als der andere Mitarbeiter Ihrer Zweiergruppe?"
    )

    contr_prob_0 = models.IntegerField(
        choices=[
            [1, 'Ja, meine Zweiergruppe kann trotzdem überprüft werden'],
            [2, 'Nein']
        ],
        label="Kann Ihre Zweiergruppe überprüft werden, auch wenn Sie beide die gleiche Anzahl an Überstunden angeben?"
    )

    contr_fix_0 = models.IntegerField(
        choices=[
            [1, '40 Lab-Punkte'],
            [2, '1 Mio. Lab-Punkte']
        ],
        label="Wie hoch ist die fixe Vergütung, die Ihnen das Unternehmen bezahlt?"
    )

    contr_var_0 = models.IntegerField(
        choices=[
            [1, '1 Lab-Punkt pro Überstunde'],
            [2, '1 Mio. Lab-Punkte']
        ],
        label="Wie hoch ist die zusätzliche Vergütung, die Ihnen das Unternehmen bezahlt?"
    )

    contr_fine_0 = models.IntegerField(
        choices=[
            [1, '0 Lab-Punkte'],
            [2, '1 Lab-Punkt pro zu viel angegebener Überstunde']
        ],
        label="Sie haben zu viele Überstunden angegeben und werden überprüft. Wie hoch ist die Strafe?"
    )

    contr_50noaudit_0 = models.IntegerField(
        label="Stellen Sie sich vor, Sie haben 20 Überstunden gemacht und geben 20 Überstunden an. Sie werden nicht überprüft. Wie hoch ist die Vergütung [in Lab-Punkten], die Ihnen das Unternehmen bezahlt?"
    )

    contr_50audit_0 = models.IntegerField(
        label="Stellen Sie sich vor, Sie haben 20 Überstunden gemacht und geben 20 Überstunden an. Sie werden überprüft. Wie hoch ist die Vergütung [in Lab-Punkten], die Ihnen das Unternehmen bezahlt?"
    )

    contr_65noaudit_0 = models.IntegerField(
        label="Stellen Sie sich vor, Sie haben 20 Überstunden gemacht und geben 30 Überstunden an. Sie werden nicht überprüft. Wie hoch ist die Vergütung [in Lab-Punkten], die Ihnen das Unternehmen bezahlt?"
    )

    contr_35audit_0 = models.IntegerField(
        label="Stellen Sie sich vor, Sie haben 20 Überstunden gemacht und geben 30 Überstunden an. Sie werden überprüft. Wie hoch ist die Vergütung [in Lab-Punkten], die Ihnen das Unternehmen bezahlt?"
    )

    # Control questions_1
    contr_partner = models.IntegerField(
        choices=[
            [1, 'Nichts'],
            [2, 'Ihre Zweiergruppe wird überprüft.']
        ],
        label="Was passiert, wenn Sie etwas anderes angeben als der andere Mitarbeiter Ihrer Zweiergruppe?"
    )

    def contr_partner_error_message(self, value):
        if value != 2:
            return 'Nicht korrekt. Bitte versuchen Sie es erneut.'


    contr_prob = models.IntegerField(
        choices=[
            [1, 'Ja, meine Zweiergruppe kann trotzdem überprüft werden'],
            [2, 'Nein']
        ],
        label="Kann Ihre Zweiergruppe überprüft werden, auch wenn Sie beide die gleiche Anzahl an Überstunden angeben?"
    )
    def contr_prob_error_message(self, value):
        if value != 1:
            return 'Nicht korrekt. Bitte versuchen Sie es erneut.'


    contr_fix = models.IntegerField(
        choices=[
            [1, '40 Lab-Punkte'],
            [2, '1 Mio. Lab-Punkte']
        ],
        label="Wie hoch ist die fixe Vergütung, die Ihnen das Unternehmen bezahlt?"
    )
    def contr_fix_error_message(self, value):
        if value != 1:
            return 'Nicht korrekt. Bitte versuchen Sie es erneut.'


    contr_var = models.IntegerField(
        choices=[
            [1, '1 Lab-Punkt pro angegeber Überstunde'],
            [2, '1 Mio. Lab-Punkte']
        ],
        label="Wie hoch ist die zusätzliche Vergütung, die Ihnen das Unternehmen bezahlt?"
    )
    def contr_var_error_message(self, value):
        if value != 1:
            return 'Nicht korrekt. Bitte versuchen Sie es erneut.'


    contr_fine = models.IntegerField(
        choices=[
            [1, '0 Lab-Punkte'],
            [2, '1 Lab-Punkt pro zu viel angegebener Überstunde']
        ],
        label="Sie haben zu viele Überstunden angegeben und werden überprüft. Wie hoch ist die Strafe?"
    )
    def contr_fine_error_message(self, value):
        if value != 2:
            return 'Nicht korrekt. Bitte versuchen Sie es erneut.'



    contr_50noaudit = models.IntegerField(
        label="Stellen Sie sich vor, Sie haben 20 Überstunden gemacht und geben 20 Überstunden an. Sie werden nicht überprüft. Wie hoch ist die Vergütung [in Lab-Punkten], die Ihnen das Unternehmen bezahlt?"
    )

    def contr_50noaudit_error_message(self, value):
        if value != 60:
            return 'Nicht korrekt. Haben Sie sowohl die fixe, als auch die zusätzliche Vergütung berücksichtigt?'

    contr_50audit = models.IntegerField(
        label="Stellen Sie sich vor, Sie haben 20 Überstunden gemacht und geben 20 Überstunden an. Sie werden überprüft. Wie hoch ist die Vergütung [in Lab-Punkten], die Ihnen das Unternehmen bezahlt?"
    )

    def contr_50audit_error_message(self, value):
        if value != 60:
            return 'Nicht korrekt. Haben Sie sowohl die fixe, als auch die zusätzliche Vergütung berücksichtigt?'

    contr_65noaudit = models.IntegerField(
        label="Stellen Sie sich vor, Sie haben 20 Überstunden gemacht und geben 30 Überstunden an. Sie werden nicht überprüft. Wie hoch ist die Vergütung [in Lab-Punkten], die Ihnen das Unternehmen bezahlt?"
    )

    def contr_65noaudit_error_message(self, value):
        if value != 70:
            return 'Nicht korrekt. Haben Sie sowohl die fixe, als auch die zusätzliche Vergütung berücksichtigt?'

    contr_35audit = models.IntegerField(
        label="Stellen Sie sich vor, Sie haben 20 Überstunden gemacht und geben 30 Überstunden an. Sie werden überprüft. Wie hoch ist die Vergütung [in Lab-Punkten], die Ihnen das Unternehmen bezahlt?"
    )

    def contr_35audit_error_message(self, value):
        if value != 50:
            return 'Nicht korrekt. Haben Sie sowohl die fixe, als auch die zusätzliche Vergütung und die Strafe berücksichtigt?'

    #############
    # payoff #
    #############

    fine = models.IntegerField()
    lab_points = models.IntegerField()
    points_euro = models.CurrencyField()
    ctr_euro = models.CurrencyField()

    def cal_payoff(self):

        if self.to_audit_player == True or self.same_hours == False:
            self.audited_both = True

        if self.audited_both and self.hours_stated > Constants.actual_surplus_hours:
            self.fine = abs(self.hours_stated - Constants.actual_surplus_hours)
        else:
            self.fine = 0

        self.lab_points = Constants.income_fix + Constants.actual_surplus_hours - self.fine if self.audited_both \
            else Constants.income_fix + self.hours_stated
        self.points_euro = self.lab_points * 0.1
        self.ctr_euro = self.control_correct * Constants.control_correct_euro

        self.payoff = self.points_euro + Constants.showup + self.ctr_euro

    #############
    # questions #
    #############
    # belief
    belief = models.IntegerField(
        min=0,
        max=100,
        label="Stellen Sie sich vor, 100 andere Personen haben ebenfalls an diesem Experiment teilgenommen. Stellen Sie sich weiterhin vor, dass diese 100 Personen jeweils 10 Überstunden geleistet haben. Wie viele dieser 100 Personen glauben Sie haben mehr als 10 Überstunden angegeben?"
    )

    # questions socio demographic
    age = models.IntegerField(label="Wie alt sind Sie in Jahren?:")
    gender = models.IntegerField(
        choices=[
            [1, 'männlich'],
            [2, 'weiblich'],
            [3, 'divers'],
        ],
        label="Sind Sie männlich, weiblich oder divers?"
    )
    study = models.IntegerField(
        choices=[
            [1, 'Katholisch / Evangelisch-Theologische Fakultät'],
            [2, 'Fachbereich Rechtswissenschaften'],
            [3, 'Fachbereich Wirtschaftswissenschaften'],
            [4, 'Medizinische Fakultät'],
            [5, 'Philosophische Fakultät'],
            [6, 'Mathematisch-Naturwissenschaftliche Fakultät'],
            [7, 'Landwirtschaftliche Fakultät'],
            [8, 'Sonstige'],
            [9, 'Ich bin kein Student'],
        ],
        label="An welcher Fakultät / an welchem Fachbereich sind Sie eingeschrieben?"
    )
    study_level = models.IntegerField(
        choices=[
            [1, 'Bachelor'],
            [2, 'Master'],
            [3, 'Diplom'],
            [4, 'Staatsexamen'],
            [5, 'Sonstige']
        ],
        label="In welchem Programm studieren Sie?"
    )

    econ = models.IntegerField(
        choices=[
            [1, 'Ja'],
            [2, 'Nein']
        ],
        label="Haben Sie mehr als eine Vorlesung aus dem Fachgebiet Wirtschaftswissenschaft besucht?"
    )

    # questions moral
    moral_freude = models.IntegerField(
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    )
    moral_aerger = models.IntegerField(
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        widget=widgets.RadioSelect
    )
    moral_angst = models.IntegerField(
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        widget=widgets.RadioSelect
    )
    moral_scham = models.IntegerField(
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        widget=widgets.RadioSelect
    )
    moral_schuld = models.IntegerField(
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        widget=widgets.RadioSelect
    )

    # questions other
    risk = models.IntegerField(
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        widget=widgets.RadioSelect
    )
    lie = models.IntegerField(
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        widget=widgets.RadioSelect
    )
    complex = models.IntegerField(
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        widget=widgets.RadioSelect
    )
    pol = models.IntegerField(
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        widget=widgets.RadioSelect
    )
    income = models.IntegerField(
        label="Wie viel Euro haben Sie im Durchschnitt nach Abzug aller anfallenden fixen Kosten (wie zum Beispiel Miete) monatlich zur freien Verfügung?",
        blank=True
    )
    pray = models.IntegerField(
        choices=[
            [1, 'Gar nicht'],
            [2, '1- bis 2-mal'],
            [3, '3- bis 5-mal'],
            [4, 'täglich'],
            [5, 'Mehrmals täglich']
        ],
        label="Wie oft beten Sie pro Woche?",
        blank=True
    )
    lab_exp = models.IntegerField(
        choices=[
            [1, 'Gar nicht'],
            [2, '1- bis 2-mal'],
            [3, '5- bis 10-mal'],
            [4, '11- bis 20-mal'],
            [5, 'mehr als 20-mal']
        ],
        label="Wie oft haben Sie bereits an ökonomischen Experimenten teilgenommen?"
    )

    # questions_expdem
    exp_demand_q = models.IntegerField(
        choices=[
            [1, 'Ja'],
            [2, 'Nein']
        ],
        widget=widgets.RadioSelectHorizontal,
        label="Haben Sie sich Gedanken gemacht, welche Forschungsfrage mit diesem Experiment untersucht werden soll?"
    )
    exp_demand_open = models.StringField(
        label="Wenn ja, welche Forschungsfrage glauben Sie, wird mit diesem Experiment untersucht?",
        blank=True
    )
    exp_demand_chat = models.IntegerField(
        choices=[
            [1, 'Ja'],
            [2, 'Nein']
        ],
        widget=widgets.RadioSelectHorizontal,
        label="Haben Sie Ihr Chat-Verhalten verändert, weil Sie vermuteten, dass die Nachrichten durch einen Dritten (z.B. den Experimentator) gelesen werden?"
    )
    exp_demand_chat_open = models.StringField(
        label="Wenn ja, inwiefern haben Sie Ihr Chat-Verhalten verändert?",
        blank=True
    )

    # outro feedback
    feedback = models.StringField(
        label="Wir würden uns freuen, wenn Sie uns ein kurzes Feedback bezüglich der Umsetzung dieses Experimentes geben könnten.Fanden Sie beispielsweise die Instruktionen oder bestimmte Fragen umständlich?",
        blank=True
    )
