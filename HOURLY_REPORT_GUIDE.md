# Hourly Factory Simulation Report Guide

This guide explains how to generate a detailed, hourly performance report for the factory simulation, with all costs displayed in Indian Rupees (₹).

## Prerequisites
Ensure yours are in the project root directory (`scalar/`).

## Terminal Command
To run the simulation and see the detailed hourly output, execute the following command in your terminal:

```bash
python3 hourly_trace.py
```

## What's in the Output?
The report will display a table for all 24 hours of the simulation, including:
- **Hour**: The time of day (00:00 to 23:00).
- **Tariff (₹)**: The electricity price at that hour, converted to Rupees.
- **Production**: Hourly production and cumulative total.
- **Cost (₹)**: Electricity costs for that hour in Rupees.
- **CO2 (kg)**: Carbon emissions for that hour.
- **Health %**: Machine health percentages for (Stamping, Molding, CNC, Compressor, Welder).
- **Status**: Any breakdown events or machine failures.

## Example Output
```text
==============================================================================================================
Hour   | Tariff (₹)   | Production           | Cost (₹)     | CO2 (kg)   | Health % (S,M,C,P,W)      | Status
--------------------------------------------------------------------------------------------------------------
00:00  | 620.84       |     989 /     989 | 1769.39      | 1.14       | [99, 99, 99, 99, 99] | OK
...
```
