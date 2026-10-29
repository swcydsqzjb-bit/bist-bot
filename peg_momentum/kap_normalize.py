"""Normalize a supported KAP report chain to one purchasing-power basis.

Initial validated chain: ASELS 2026/09. This is not a full-market adapter.
PDF note coefficients and share counts are required; unknown chains fail closed.
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pandas as pd

from .financial_source import get_text, discover, PAGE
from .engine import fundamental

MONTHS = {1:'Ocak',2:'Şubat',3:'Mart',4:'Nisan',5:'Mayıs',6:'Haziran',7:'Temmuz',8:'Ağustos',9:'Eylül',10:'Ekim',11:'Kasım',12:'Aralık'}
TAGS = dict(profit='ifrs-full_ProfitLossAttributableToOwnersOfParent',
            sales='ifrs-full_Revenue', operating='ifrs-full_ProfitLossFromOperatingActivities',
            cashflow='ifrs-full_CashFlowsFromUsedInOperatingActivities',
            depreciation='ifrs-full_AdjustmentsForDepreciationAndAmortisationExpense',
            capital='ifrs-full_IssuedCapital', cash='ifrs-full_CashAndCashEquivalents',
            short_debt='kap-fr_CurrentBorowings', current_long_debt='kap-fr_CurrentPortionOfNoncurrentBorrowings',
            long_debt='ifrs-full_LongtermBorrowings')
TAGS['assets']='ifrs-full_Assets'
CHAINS = {'ASELS': {'current':1688432,'annual':1561039,'prior':1511313,'end':'2026-09-30'}}


def number(text):
    value=text.strip().replace('\u00a0','').replace(' ','')
    if not value or value=='-':
        return None
    negative=value.startswith('(') and value.endswith(')')
    value=value.strip('()').replace('.','').replace(',','.')
    result=float(value)
    if not pd.notna(result) or abs(result)==float('inf'):
        raise ValueError('invalid_numeric_value')
    return -result if negative else result


def parse_report(html, report_id):
    soup=BeautifulSoup(html,'html.parser')
    match=re.search(r'Gönderim Tarihi\s+(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2})',soup.get_text(' ',strip=True))
    if not match:
        raise ValueError('publication_time_missing')
    published=datetime.strptime(match[1],'%d.%m.%Y %H:%M:%S').replace(tzinfo=ZoneInfo('Europe/Istanbul'))
    statements={}
    for table in soup.select('table.financial-table'):
        headers=table.select(':scope > tbody > tr > td.context-header')
        if not headers:
            continue
        contexts=[]
        for h in headers:
            tr=h.select_one('.content-tr')
            dates=re.findall(r'\d{2}\.\d{2}\.\d{4}',tr.get_text(' ',strip=True) if tr else '')
            contexts.append([datetime.strptime(d,'%d.%m.%Y').date().isoformat() for d in dates])
        unit_table=table.find_previous('table',class_='financial-header-table')
        unit_match=re.search(r'(1(?:\.000){0,2})\s*TL',unit_table.get_text(' ',strip=True) if unit_table else '')
        if not unit_match:
            raise ValueError('TRY_presentation_unit_missing')
        unit=int(unit_match[1].replace('.',''))
        values={}
        for row in table.select(':scope > tbody > tr'):
            name=row.select_one(':scope > td.taxonomy-field-name-cell .taxonomy-field-name')
            cells=row.select(':scope > td.taxonomy-context-value')
            if name is None or not cells:
                continue
            tag=name.get_text(strip=True).split('|')[0]
            if tag not in TAGS.values():
                continue
            if tag in values or len(cells)!=len(contexts):
                raise ValueError('ambiguous_statement_context')
            values[tag]=[None if number(c.get_text(strip=True)) is None else number(c.get_text(strip=True))*unit for c in cells]
        kind=('income' if TAGS['sales'] in values else 'cashflow' if TAGS['cashflow'] in values else 'balance' if TAGS['capital'] in values else None)
        if kind:
            if kind in statements:
                raise ValueError('duplicate_statement_type')
            statements[kind]=dict(contexts=contexts,values=values,unit=unit)
    if set(statements)!={'income','balance','cashflow'}:
        raise ValueError('required_statements_missing')
    pdfs=list(dict.fromkeys(a['href'] for a in soup.find_all('a',href=True) if '/tr/api/file/download/' in a['href'] and a.get_text(strip=True).lower().endswith('.pdf')))
    if not pdfs:
        raise ValueError('report_pdf_missing')
    return dict(report_id=report_id,published=published,statements=statements,pdf_url=pdfs[0])


def coefficients(text, end):
    # A reference to TMS29 for subsidiaries is not enough to classify a foreign
    # functional-currency group as a TRY inflation-accounting group.
    if re.search(r'(?:geçerli|fonksiyonel) para birimi.{0,180}ABD Dolar',text,re.S|re.I):
        raise ValueError('foreign_functional_currency_requires_separate_model')
    marker=re.search(r'Dönem\s+Endeks\s+Düzeltme Katsayısı',text)
    if marker is None:
        raise ValueError('explicit_inflation_coefficient_table_missing')
    block=text[marker.end():marker.end()+900]
    end=pd.Timestamp(end)
    dates=[end,pd.Timestamp(end.year-1,12,31),end-pd.DateOffset(years=1)]
    result={}
    for d in dates:
        pattern=rf'{d.day}\s+{MONTHS[d.month]}\s+{d.year}\s+([\d.,]+)\s+([\d.,]+)'
        m=re.search(pattern,block)
        if not m:
            raise ValueError('required_reported_coefficient_missing')
        factor=number(m[2])
        if factor is None or factor<=0:
            raise ValueError('invalid_coefficient')
        result[str(d.date())]=factor
    if result[str(end.date())]!=1:
        raise ValueError('coefficient_target_is_not_current_period')
    return result


def share_count(text):
    # Explicit weighted-share note in the supported chain; capital alone does
    # not establish the weighted EPS denominator or eliminate dilution.
    m=re.search(r'Adi hisse senetleri\s*\(Bin adet\)\s+([^\n]+)',text)
    if not m:
        raise ValueError('weighted_share_note_missing')
    counts=[number(v)*1000 for v in re.findall(r'\d[\d.]*',m[1])]
    if len(counts)<2 or min(counts)<=0 or max(counts)!=min(counts):
        raise ValueError('changing_weighted_shares_requires_adjustment')
    if not re.search(r'sulandırılmış hissesi bulunmamaktadır',text,re.I):
        raise ValueError('dilution_status_unknown')
    return counts[0]


def item(report,kind,key,col):
    values=report['statements'][kind]['values'].get(TAGS[key])
    if values is None or len(values)<=col or values[col] is None:
        raise ValueError('missing_'+key)
    return values[col]


def ttm(current_ytd,annual,previous_ytd,annual_factor):
    return current_ytd + annual*annual_factor - previous_ytd


def build(current,annual,prior,pdfs,price,valuation_date,asof):
    end=current['statements']['income']['contexts'][0][-1]
    target=pd.Timestamp(end)
    if target.month not in {3,6,9}:
        raise ValueError('only_supported_interim_chains')
    expected_current=[f'{target.year}-01-01',end]
    prior_end=str((target-pd.DateOffset(years=1)).date())
    if current['statements']['income']['contexts'][:2]!=[expected_current,[f'{target.year-1}-01-01',prior_end]]:
        raise ValueError('current_context_mismatch')
    if prior['statements']['income']['contexts'][:2]!=[[f'{target.year-1}-01-01',prior_end],[f'{target.year-2}-01-01',str((target-pd.DateOffset(years=2)).date())]]:
        raise ValueError('prior_context_mismatch')
    if annual['statements']['income']['contexts'][:2]!=[[f'{target.year-1}-01-01',f'{target.year-1}-12-31'],[f'{target.year-2}-01-01',f'{target.year-2}-12-31']]:
        raise ValueError('annual_context_mismatch')
    # All reports must have been public before the chosen completed-session date.
    cutoff=datetime.combine(pd.Timestamp(asof).date(),datetime.strptime('18:00','%H:%M').time(),ZoneInfo('Europe/Istanbul'))
    if any(r['published']>cutoff for r in [current,annual,prior]):
        raise ValueError('future_report')
    for r in [current,annual,prior]:
        inc=r['statements']['income']['contexts'][:2]
        if r['statements']['cashflow']['contexts'][:2]!=inc:
            raise ValueError('cashflow_income_context_mismatch')
    for report,text in zip([current,annual,prior],pdfs):
        date=pd.Timestamp(report['statements']['income']['contexts'][0][-1])
        pattern=rf'{date.day}\s+{MONTHS[date.month]}\s+{date.year}.{{0,80}}satın alma gücü'
        if not re.search(pattern,text,re.I):
            raise ValueError('PDF_purchasing_power_date_not_verified')
    factors=coefficients(pdfs[0],end)
    shares=[share_count(text) for text in pdfs]
    if len(set(shares))!=1:
        raise ValueError('cross_report_share_count_changed')
    fy_factor=factors[f'{target.year-1}-12-31']
    old_factor=factors[prior_end]
    def bridge(a,b):
        if abs(a-b)>max(2000,abs(b)*.00015):
            raise ValueError('cross_report_restatement_bridge_mismatch')
    # Reconcile repeated observations across reports. A mere date match must not
    # silently accept later accounting-policy/restatement differences.
    bridge(item(current,'income','profit',1),item(prior,'income','profit',0)*old_factor)
    bridge(item(current,'income','sales',1),item(prior,'income','sales',0)*old_factor)
    bridge(item(current,'cashflow','cashflow',1),item(prior,'cashflow','cashflow',0)*old_factor)
    bridge(item(current,'balance','assets',1),item(annual,'balance','assets',0)*fy_factor)
    bridge(item(annual,'balance','assets',1)*fy_factor,item(prior,'balance','assets',1)*old_factor)
    profit=ttm(item(current,'income','profit',0),item(annual,'income','profit',0),item(current,'income','profit',1),fy_factor)
    previous=ttm(item(current,'income','profit',1),item(annual,'income','profit',1)*fy_factor,item(prior,'income','profit',1)*old_factor,1)
    cash=ttm(item(current,'cashflow','cashflow',0),item(annual,'cashflow','cashflow',0),item(current,'cashflow','cashflow',1),fy_factor)
    operating=ttm(item(current,'income','operating',0),item(annual,'income','operating',0),item(current,'income','operating',1),fy_factor)
    depreciation=ttm(item(current,'cashflow','depreciation',0),item(annual,'cashflow','depreciation',0),item(current,'cashflow','depreciation',1),fy_factor)
    # Use the report's explicit quarterly columns; never subtract incompatible YTDs.
    contexts=current['statements']['income']['contexts']
    start=str((target-pd.DateOffset(months=2)).replace(day=1).date())
    old_start=str((pd.Timestamp(start)-pd.DateOffset(years=1)).date())
    if contexts[2:]!=[[start,end],[old_start,prior_end]]:
        raise ValueError('explicit_quarterly_contexts_missing')
    net_debt=sum(item(current,'balance',k,0) for k in ['short_debt','current_long_debt','long_debt'])-item(current,'balance','cash',0)
    eps=profit/shares[0]
    if eps<=0:
        raise ValueError('nonpositive_ttm_eps')
    available=pd.Timestamp(current['published'].date())
    if current['published'].hour>=18:
        available+=pd.offsets.BDay(1)
    row=dict(symbol='ASELS',sector_model='nonfinancial',source=';'.join('https://www.kap.org.tr/tr/Bildirim/'+str(r['report_id']) for r in [current,annual,prior]),
             period_end=end,published_at=str(available.date()),valuation_date=valuation_date,
             comparison_basis=f'TRY TMS29 at {end}; coefficients as reported, rounded to 4 decimals; stable weighted shares verified in all 3 PDFs',
             comparable=True,recurring_earnings_verified=False,pe_ttm=price/eps,eps_ttm=eps,eps_prior_ttm=previous/shares[0],
             profit_ttm=profit,profit_prior_ttm=previous,operating_profit_q=item(current,'income','operating',2),
             operating_profit_prior_year_q=item(current,'income','operating',3),sales_q=item(current,'income','sales',2),
             sales_prior_year_q=item(current,'income','sales',3),operating_cashflow_ttm=cash,net_debt=net_debt,ebitda_ttm=operating+depreciation)
    growth=(profit/previous-1)*100 if previous>0 else None
    audit=dict(period_end=end,coefficients=factors,weighted_shares=shares[0],
               eps_growth_pct=growth,peg_historical=row['pe_ttm']/growth if growth and growth>0 else None,
               ebitda_method='operating profit plus cashflow depreciation; proxy, not company-adjusted EBITDA',
               recurring_earnings_verified=False,scope='ASELS_CHAIN_ONLY',
               report_ids=[r['report_id'] for r in [current,annual,prior]])
    return row,audit


def pdf_text(url,cache):
    name=hashlib.sha256(url.encode()).hexdigest()
    pdf=cache/(name+'.pdf')
    text=cache/(name+'.txt')
    if not pdf.exists():
        with urlopen(url,timeout=40) as r:
            pdf.write_bytes(r.read())
    subprocess.run(['pdftotext','-layout',str(pdf),str(text)],check=True,capture_output=True)
    return text.read_text()


def collect(output='peg_momentum_output',cache_path='.kap_report_cache',asof=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cache=Path(cache_path);cache.mkdir(parents=True,exist_ok=True)
    chain=CHAINS['ASELS']; reports=[];texts=[]
    # Do not keep using a frozen manifest after the provider lists a new report.
    _,periods=discover(get_text(PAGE+'?hisse=ASELS'),datetime.now(ZoneInfo('Europe/Istanbul')).date())
    if periods[0]!=(2026,9):
        raise ValueError('validated_chain_no_longer_latest')
    for name in ['current','annual','prior']:
        rid=chain[name];file=cache/(str(rid)+'.json')
        if file.exists():
            report=json.loads(file.read_text())
            report['published']=datetime.fromisoformat(report['published'])
        else:
            report=parse_report(get_text('https://www.kap.org.tr/tr/Bildirim/'+str(rid)),rid)
            stored=dict(report,published=report['published'].isoformat())
            temp=file.with_suffix('.tmp')
            temp.write_text(json.dumps(stored,ensure_ascii=False),encoding='utf-8')
            temp.replace(file)
        reports.append(report)
        texts.append(pdf_text(report['pdf_url'],cache))
    import yfinance as yf
    from .run import completed_day
    asof=pd.Timestamp(asof) if asof is not None else completed_day()
    prices=yf.Ticker('ASELS.IS').history(period='1mo',auto_adjust=False,timeout=25)
    if prices.empty:
        raise ValueError('price_missing')
    index=pd.DatetimeIndex(prices.index)
    if index.tz is not None:
        index=index.tz_convert('Europe/Istanbul').tz_localize(None)
    prices.index=index.normalize();prices=prices[prices.index<=asof]
    if prices.empty:
        raise ValueError('completed_price_missing')
    valued=str(prices.index[-1].date())
    row,audit=build(*reports,texts,float(prices.Close.iloc[-1]),valued,asof)
    status,reasons,metrics=fundamental(row,asof)
    audit.update(asof=str(asof.date()),price_date=valued,price=float(prices.Close.iloc[-1]),status=status,reasons=reasons,metrics=metrics)
    pd.DataFrame([row]).to_csv(out/'kap_normalized_preview.csv',index=False)
    (out/'kap_normalization_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    return out/'kap_normalized_preview.csv'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='peg_momentum_output')
    p.add_argument('--cache',default='.kap_report_cache')
    args=p.parse_args()
    collect(args.output,args.cache)


if __name__=='__main__':
    main()
