# IMPORTANT: Year Mapping for 10-K Filings

## Understanding the Data

The assignment PDFs are named with **filing years**, not **fiscal years**:

| Filing Year | PDF Filename | Contains Fiscal Years | Key Insight |
|-------------|--------------|----------------------|-------------|
| **2020** | `*-20-*.pdf` | 2017, 2018, 2019 | To get 2019 data, use year=2020 |
| **2021** | `*-21-*.pdf` | 2018, 2019, **2020** | To get 2020 data, use year=2021 |
| **2022** | `*-22-*.pdf` | 2019, 2020, **2021** | To get 2021 data, use year=2022 |

## Examples

### To find Uber's 2020 revenue:
```python
# WRONG: Searches 2020 filing (has 2019 data)
result = rag_agent.answer("What was Uber's revenue in 2020?", company="UBER", year=2020)

# CORRECT: Searches 2021 filing (has 2020 data)
result = rag_agent.answer("What was Uber's revenue in 2020?", company="UBER", year=2021)
```

### To find Amazon's 2019 revenue:
```python
# Use 2020 filing (contains 2019 data)
result = rag_agent.answer("What was Amazon's revenue in 2019?", company="AMZN", year=2020)
```

## Quick Reference

| Desired Fiscal Year | Use year= | Because |
|---------------------|-----------|---------|
| 2019 | 2020 | 2020 10-K contains FY2019 |
| 2020 | 2021 | 2021 10-K contains FY2020 |
| 2021 | 2022 | 2022 10-K contains FY2021 |

## How the System Works

The `year` parameter in the code refers to the **filing year** (from the PDF filename), not the fiscal year you're looking for.

```python
# In pdf_loader.py:
year = 2000 + year_suffix  # Extracts -20-, -21-, -22- from filename
```

This is why when you search for "revenue in 2020" with year=2020, you get 2019 data - because the 2020 filing contains 2019 fiscal data!
