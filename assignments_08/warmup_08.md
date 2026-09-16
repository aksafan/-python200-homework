# --- Cloud Concepts ---

## Q1
> What is the core economic model of cloud computing, and how does it differ from owning your own servers?

You're renting compute resources from a cloud provider, and paying for resources on demand / pay-as-you-go.
Owning a bare metal servers requires full control and maintenance of both hardware and software.

## Q2
> What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Vertical scaling is the way to add more compute resources (CPU, GPU, Memory) to let an app to handle the load. Usually, a great way to react proactively on some rapid changes in load (e.g. a marketing campaign).
Horizontal scaling is the way to add more servers/machines and split the load in between them. Usually, a great way to scale a huge DB that doesn't fit into one instance or became too slow cause of way too much data.

> Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

> A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.

Horizontal scaling, cause this is the fastest and simplest way to hold the load.

> A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.

Vertical scaling, cause this is what vertical scaling does - adding after GPU and more RAM/CPU.

> A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.

Horizontal scaling, cause the work can be split and it will be much faster to do that on multiple machines.

## Q3
> Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.

- Gmail - SaaS, fully build app for us to use without any bothering about infrastructure.
- Azure Virtual Machines - IaaS, proposed a cloud computing resources with access to all configuration starting from OS and for all the code.
- AWS S3 (Simple Storage Service) - IaaS, it is a raw infrastructure to be configured and connect to your application.
- GitHub Codespaces - PaaS, an application to use with some control over it, still with a full service package.
- Snowflake - SaaS, fully managed by their developers, no hustle with an infrastructure full out of the box functionality.
- Supabase - BaaS, it gives you a fully managed database and backend, you just add a code and can call APIs or query DB.

> Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.

- IaaS - gives access to a cloud instance with an OS, and that's it. I'm responsible for managing everything from configuration to the code. AWS EC2 is a great example.
- PaaS - gives an access to a managed infrastructure, and I'm adding only the code. GitHub Codespaces will be an example.
- SaaS - gives access to the whole application functionality out of the box (e.g. Gmail, Outlook, Jira, etc.)

## Q4
> What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?

A managed data platform is a SaaS that runs over cloud servers and wraps all that internal things, so users can just use the functionality via a convenient interface. It differs from using cloud providers in the way how you manage the platform: for data platform you don't need knowledge and spent time for managing infrastructure, you can just use optimized engineers and data tools, have everything autoscaled, and start to work without additional setup.
Still, with all the pros, there are cons:
- Pricing is usually bigger
- Less control over the system
- Vendor lock and platform switch costs


## Q5
> The lesson a situation where the cloud is probably not the right choice. What is that situation?

I can think of robotics when a response time is crucial and having everything on cloud will slow things up for a mile. Also, financial trading for the same reason.
Another close case can be due to a legal reasons and compliance. E.g. to support GDPR we need to make sure data is stored inside EU.


# --- Cloud Landscape ---

## Q1
> Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.

- AWS - the oldest, the biggest variety of services, has everything you can imagine. Usually used by companies historically have been using it for long time, and have engineers familiar with it.
- GCP - best for ML and data work. Used by companies who have data intense tasks.
- Azure - best for government and enterprise companies cause of the integration with other MS tools. Used by companies in government sector and huge enterprises.

## Q2
> The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.

- Easy setup, especially for newcomers
- SQL usage and convenience of learning one thing and using the knowledge to others
- Great fit for data models concepts training

> Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?

Spend some time on setup and understanding of available tools, get used to main tables, think about reiterating on data modeling concepts based on real work.


## Q3
> For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.

> You need to store 10 TB of image files and retrieve them by filename from any machine.

Object/blob storage, AWS S3

> You need to run an ML training job on a GPU for four hours, then shut it down.

ML platform (cloud compute / GPU compute / ML compute), GCP Vertex AI

> You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.

Serverless compute, AWS Lambda

> You need to send structured data to a large language model and get a text response back.

LLM API, AWS Bedrock

## Q4

> The lesson says most projects don't use one provider for everything. Describe a simple data project of your own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, and what would you give up if you did?

E-commerce analytics tool. A dashboard to show sales.
We can use BigQuery as a storage and query engine, S3 as a bloc storage for big files to process or reports storing, and a serverless AWS lambdas to send ready to use data via API.
Consolidating might help with overall billing, as it should be cheaper to have everything under for example AWS, less complexity and learning curve, the only one SDK to use.
What I'd give up are a vendor lock to one vendor and losing ability to chose the best solution for each case.
