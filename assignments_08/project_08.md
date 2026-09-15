# Part A: Supabase Setup
My Supabase project is set up, everything works as expected.

# Part B: Cloud Cost Analysis
> Before recording your video, build estimates for the two scenarios below, then feel free to explore further. There are hundreds of services — throw in whatever looks interesting and see what happens to the total.

> Scenario A — Lightweight compute: A t3.micro EC2 instance (1 vCPU, 1 GB RAM), on-demand pricing, running 8 hours per day, 5 days per week (approximately 160 hours per month). Use the US East (N. Virginia) region.

> Scenario B — Heavy analytics workload: A p3.2xlarge EC2 instance (8 vCPU, 1 V100 GPU), running 24/7 for the full month (730 hours); an RDS db.m5.large instance (2 vCPU, 8 GB RAM); and an S3 Standard storage bucket with 1 TB of data. Use US East (N. Virginia).

> In project_08.md, write a short summary (a few sentences to a paragraph) covering:

> - What each scenario costs, and whether the numbers surprised you.

- Scenario A — Lightweight compute costs:
  - Monthly - 1.66 USD
  - Annually - 19.92 USD
- Scenario B — Heavy analytics workload costs:
  - Monthly - 2,580.38 USD
  - Annually - 30,964.56 USD 

Something that surprised me was the huge difference between the two scenarios. Scenario A is very cheap, while Scenario B is extremely expensive.
I expected Scenario B to be more expensive, not to that degree - that one is just insanely high.

> - Anything interesting you found while exploring the calculator beyond the two required scenarios.

The amount of different services in general and options inside each service in particular.
There were so many of them that I was a bit lost at first.

> - A sentence on how the two scenarios compare — what does the cost difference tell you about when a GPU instance is or isn't worth it?

The cost difference between the two scenarios is just huge! From my understanding GPU instances are the most expensive ones, and they are worth only when I need to do some heavy ML work or parallel computations that require a lot of processing power in general.

# The Video

https://youtu.be/DoNhxixIEuE
