# Requirements

- Python >= 3.8
- as of today, only edf files by EyeLink eye trackers are supported

If you want to use eye2bids to convert EyeLink data,
you will need to install the EyeLink Developers Kit.
It can be downloaded from [SR-Research support forum](https://www.sr-research.com/support/).

!!! info "EyeLink Developers Kit"

    SR-Research support forum registration required. Register and download *EyeLink Developers Kit / API*.

The [installation on Ubuntu](https://www.sr-research.com/support/docs.php?topic=linuxsoftware) can also be done with the following commands:

```bash
sudo add-apt-repository universe
sudo apt update
sudo apt install ca-certificates
sudo apt-key adv --fetch-keys https://apt.sr-research.com/SRResearch_key
sudo add-apt-repository 'deb [arch=amd64] https://apt.sr-research.com SRResearch main'
sudo apt update
sudo apt install eyelink-display-software
```
