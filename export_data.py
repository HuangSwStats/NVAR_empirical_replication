"""Re-extract the packaged ten-country CSV data from the included workbook."""
from pathlib import Path
import json, hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parent

def main():
    book=ROOT/'data/source_workbook.xlsx'
    panels=[pd.read_excel(book,sheet_name=s,index_col=0).dropna(how='all').dropna(axis=1,how='all')
            for s in ('Trade转置N=10','REER转置N=10')]
    trade,reer=panels
    assert list(trade.index)==list(reer.index)
    dates=[d for d in trade.columns if d in reer.columns]
    for panel,name in zip(panels,['trade_levels.csv','reer_levels.csv']):
        out=panel[dates].T
        out.index.name='month'
        assert out.shape==(294,10) and not out.isna().any().any()
        out.to_csv(ROOT/'data'/name,float_format='%.17g')
    metadata=dict(workbook_sha256=hashlib.sha256(book.read_bytes()).hexdigest(),
        sheets=['Trade转置N=10','REER转置N=10'],countries=list(trade.index),
        period=[str(dates[0]),str(dates[-1])],months=len(dates),
        input_status='Already-standardized levels supplied in source workbook, not raw WTO/Wind downloads.',
        transformation='Intersect months, first differences, country-specific month-of-year demeaning, full-sample centering and sample-standard-deviation scaling (ddof=1).',
        csv_sha256={name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
                    for name in ['trade_levels.csv','reer_levels.csv']})
    (ROOT/'data/metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')

if __name__=='__main__': main()
