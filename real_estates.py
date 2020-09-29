import pandas
import numpy
import numpy_financial
import datetime
import matplotlib
import copy
from enum import Enum


class TaxSystem(Enum):
    Real = 1
    Micro = 2


class Options:
    @property
    def acquisition_cost(self):
        return self.notary_cost + self.bank_cost

    @property
    def property_value(self):
        return self.initial_property_value + self.initial_renovation_cost * 0.7

    def __init__(self):
        # Property
        self.initial_property_value = 0
        self.property_value_increase_rate = 1/100

        # Income
        self.rental_income = 0
        self.rental_occupancy_rate = 11/12
        self.income_tax_rate = 47.2/100

        # Acquisition Cost
        self.notary_cost = None
        self.bank_cost = None

        # renovation cost
        self.initial_renovation_cost = 0
        self.yearly_renovation_cost = None

        # Initial cash injection
        self.initial_cash_injection = 0

        # loan
        self.loan_amount = None
        self.loan_rate = 1.6/100
        self.loan_duration = 12 * 21

        # yearly
        self.property_tax = None
        self.common_maintenance_cost = None
        self.insurance_cost = None

        # other cost
        self.additional_cost = {}

        # other amortization
        self.additional_amortization = {}

        # Tax Subsidy
        self.tax_subsidy = []

        self.tax_system = TaxSystem.Real
        self.tax_reduction = 0/100

        self.st = datetime.datetime.today
        self.simulation_duration = None

    def set_defaults(self):
        if self.property_tax is None:
            self.property_tax = 0.75 * self.rental_income
        if self.common_maintenance_cost is None:
            self.common_maintenance_cost = 0.6 * self.rental_income
        if self.insurance_cost is None:
            self.insurance_cost = 0.1 * self.rental_income
        if self.notary_cost is None:
            self.notary_cost = 8/100 * self.initial_property_value
        if self.bank_cost is None:
            self.bank_cost = 0.015 * self.loan_amount
        if self.yearly_renovation_cost is None:
            self.yearly_renovation_cost = 0.5 * self.rental_income
        if self.simulation_duration is None:
            self.simulation_duration = self.loan_duration + 12
        if self.loan_amount is None:
            self.loan_amount = self.initial_property_value + \
                self.initial_renovation_cost + self.acquisition_cost - self.initial_cash_injection

    def override(self, **args):
        c = copy.deepcopy(self)
        for arg in args:
            c.__dict__[arg] = args[arg]
        return c

    def pinel(self):
        pinel_flow = [2/100 * self.initial_property_value for i in range(
            9)] + [1/100 * self.initial_property_value for i in range(3)]
        pinel_flow = [pinel_flow[int(i/12)] if i %
                      12 == 0 else 0 for i in range(12 * len(pinel_flow))]
        return pinel_flow


#compute_loan_values(pandas.DataFrame({'_': [0 for i in range(25)]}), 263750, 1.6/100, 12*20)
def compute_loan_values(df, capital, rate, duration):
    loan_monlthy_payment = (capital * rate/12) / (1 - (1 + rate/12)**-duration)

    df['Loan Capital'] = capital
    df['Loan Payment'] = loan_monlthy_payment
    df['Loan Interest'] = 0

    for i in df.index:
        if i > 0:
            df.loc[i, 'Loan Capital'] = df.loc[i-1, 'Loan Capital'] - \
                df.loc[i-1, 'Loan amortisation']
        df.loc[i, 'Loan Interest'] = rate/12 * df.loc[i, 'Loan Capital']
        df.loc[i, 'Loan amortisation'] = df.loc[i,
                                                'Loan Payment'] - df.loc[i, 'Loan Interest']

        if df.loc[i, 'Loan Capital'] <= 0:
            df.loc[i, 'Loan Capital'] = 0
            df.loc[i, 'Loan Interest'] = 0
            df.loc[i, 'Loan amortisation'] = 0
            df.loc[i, 'Loan Payment'] = 0

    return df


def compute_property_value(df, property_value, rate):
    df['Rate'] = df.apply(lambda x: (1+rate) if x.name %
                          12 == 0 else 1, axis=1)
    df.loc[0, 'Rate'] = 1
    df['Rate'] = df['Rate'].cumprod()
    df['Property value'] = property_value * df['Rate']
    df = df.drop('Rate', axis=1)
    return df


def compute_rental_income(df, income, occupancy):
    df['Income'] = income * occupancy
    return df


#df = pandas.DataFrame({'Income': [1000, 1000, 1000, 1000, 1000, 1000], 'Cost': [2000, 500, 500, 500, 500, 500]})
#compute_income_tax(df, 47.2/100, 50)
def compute_income_tax(df, rate, amort=0):
    df['Taxable'] = 0
    df['Deficit'] = 0
    deficit = 0

    for i in df.index:
        taxable = df.loc[i, 'Income'] - df.loc[i, 'Cost'] - deficit - amort

        if taxable >= 0:
            df.loc[i, 'Taxable'] = taxable
            deficit = 0

        if taxable < 0:
            deficit = -taxable

        df.loc[i, 'Deficit'] = deficit

    df['Tax'] = rate * df['Taxable']
    return df


def compute_income_tax_micro(df, rate, reductio_rate):
    df['Taxable'] = 0
    df['Deficit'] = 0
    deficit = 0

    for i in df.index:
        taxable = df.loc[i, 'Income']
        if taxable >= 0:
            df.loc[i, 'Taxable'] = taxable

    df['Tax'] = rate * (1 - reductio_rate) * df['Taxable']
    return df


def compute_irr(df, initial_invest=0):
    irr = 0
    computed_irr = []
    yearly_cash_flow = [df['Cash Flow'].iloc[i*12:i*12+12].sum()
                        for i in range(int(len(df) / 12))]

    for i in range(len(yearly_cash_flow)):
        cf = list(yearly_cash_flow[:i])
        cf += [df['Resell value'].iloc[i*12]]
        cf[0] = cf[0] - initial_invest

#        if irr == 0 or i % 6 == 0:
        irr = numpy_financial.irr(cf) * 100
        computed_irr += [irr]

    df['IRR'] = [computed_irr[int((i-i % 12)/12)] for i in range(len(df))]
    return df


def simulation_base(opt: Options):
    df = pandas.DataFrame()
    df['Month'] = [i+1 for i in range(opt.simulation_duration)]
    df['Date'] = [datetime.datetime(year=opt.st.year+int(((opt.st.month+m)-(
        opt.st.month+m) % 12)/12), month=(opt.st.month+m) % 12+1, day=1) for m in df['Month']]
    df = df.drop('Month', axis=1)

    df = compute_property_value(
        df, opt.property_value, opt.property_value_increase_rate)
    df = compute_rental_income(
        df, opt.rental_income, opt.rental_occupancy_rate)
    df = compute_loan_values(
        df, opt.loan_amount, opt.loan_rate, opt.loan_duration)

    df['Acquisition Cost'] = [opt.acquisition_cost if i ==
                              0 else 0 for i in range(opt.simulation_duration)]
    df['Renovation Cost'] = [opt.initial_renovation_cost if i ==
                             0 else opt.yearly_renovation_cost/12 for i in range(opt.simulation_duration)]

    df['Property Tax'] = opt.property_tax / 12
    df['Insurance Cost'] = opt.insurance_cost / 12
    df['Common Maintenance Cost'] = opt.common_maintenance_cost / 12

    df['Additional Cost'] = numpy.sum(
        [opt.additional_cost[c] for c in opt.additional_cost])

    df['Amortization'] = numpy.sum(
        [opt.additional_amortization[c] for c in opt.additional_amortization])

    df['Tax Subsidy'] = 0
    for i, v in enumerate(opt.tax_subsidy):
        df.loc[i, 'Tax Subsidy'] = v

    df.index = df['Date']
    df = df.drop('Date', axis=1)

    return df


def compute(opt: Options):
    df = simulation_base(opt)

    df['Cost'] = 0
    df['Cost'] += df['Additional Cost']
    df['Cost'] += df['Loan Interest']
    df['Cost'] += df['Property Tax']
    df['Cost'] += df['Common Maintenance Cost']
    df['Cost'] += df['Insurance Cost']
    df['Cost'] += df['Renovation Cost']

    if opt.tax_system == TaxSystem.Real:
        df = compute_income_tax(df, opt.income_tax_rate,
                                amort=df['Amortization'].iloc[0])
    if opt.tax_system == TaxSystem.Micro:
        df = compute_income_tax_micro(
            df, opt.income_tax_rate, opt.tax_reduction)

    df['Cash Flow'] = df['Income'] + df['Tax Subsidy'] - df['Cost'] - df['Acquisition Cost'] - \
        df['Loan amortisation'] - df['Tax']
    df['Treasury'] = df['Cash Flow'].cumsum()
    df['Resell value'] = df['Property value'] - df['Loan Capital']
    df['NAV'] = df['Resell value'] + df['Treasury']

    df = compute_irr(df, opt.initial_cash_injection)
    return df


def summary_one(name, df):
    result = pandas.DataFrame()
    result['Duree'] = [int(len(df[df['Loan Interest'] > 0])/12)]
    result['Cash Flow'] = [int(df['Cash Flow'].sum() /
                               len(df[df['Loan Interest'] > 0]))]
    result['Mensualite pret'] = [-int(df['Loan Payment'].sum() /
                                      len(df[df['Loan Interest'] > 0]))]
    result['Total Interet'] = [-int(df['Loan Interest'].sum())]
    result['Total Loyer'] = [int(df['Income'].sum())]
    result['Total IR'] = [-int(df['Tax'].sum())]
    result['Total Others Fees'] = [- int(df['Cost'].sum())
                                   - int(df['Acquisition Cost'].sum())
                                   - result['Total Interet'][0]]
    result['NAV'] = [int(df['NAV'].iloc[-1])]
    result['IRR'] = [(df['IRR'].iloc[-1])]
    result.index = [name]
    return result


def summary(simulations):
    return pandas.concat([summary_one(k, simulations[k]) for k in simulations])


def effort_epargne_one(df, title):
    df = df[df['Loan Interest'] > 0]

    sub = df['Tax Subsidy'].sum()
    ee = df[df['Cash Flow'] < 0]['Cash Flow'].sum()
    loc = df[df['Loan Capital'] > 0]['Income'].sum()
    resell = df['Resell value'].iloc[-1]

    x = pandas.DataFrame()
    if sub > 0:
        x['Label'] = ['Effort Epargne', 'Locataire', 'Defiscalisation']
        x['Financement'] = [-ee, loc, sub]
    else:
        x['Label'] = ['Effort Epargne', 'Locataire']
        x['Financement'] = [-ee, loc]
    x.index = x['Label']
    x = x.drop('Label', axis=1)
    x.plot(kind='pie', y='Financement', title=title)


def effort_epargne(simulations):
    for s in simulations:
        effort_epargne_one(simulations[s], s)


def plot_cost_one(df, title):
    df = df[df['Loan Interest'] > 0]
    l = df['Loan Interest'].sum()
    pt = df['Property Tax'].sum()
    t = pt + df['Tax'].sum()
    cmc = df['Common Maintenance Cost'].sum()
    other = df['Additional Cost'].sum()
    other += df['Cost'].sum() - l - pt - cmc - other

    cost = pandas.DataFrame()
    cost['Label'] = ['Loan', 'Tax',
                     'Common Maintenance Cost', 'Others']
    cost['Cost'] = [l, t, cmc, other]

    cost.index = cost['Label']
    cost = cost.drop('Label', axis=1)
    cost.plot(kind='pie', y='Cost',
              title=f"Cost repartition {title} {int(cost['Cost'].sum())} €")


def plot_cost(simulations):
    for s in simulations:
        plot_cost_one(simulations[s], s)


def plot_irr(d, date):
    irr_df = pandas.DataFrame()
    for k in d:
        irr_df[k] = d[k]['IRR']
        irr_df = irr_df[irr_df.index > date]
    irr_df.plot()
