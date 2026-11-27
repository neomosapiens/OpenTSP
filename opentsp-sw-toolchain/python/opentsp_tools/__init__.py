"""opentsp_tools: the scheduling model and the program-text tools a producer of OpenTSP programs uses.

  profile   a profile (config/<name>.json of the specification) and the encoding database
  stations  station gaps and the equation that meets a producer and a consumer (ScheduleError)
  sched     the tracker of stream, slot and bank use that a scheduler checks its program against
  queues    queue names, the references of an AGT loop
  emit      `.ots` text: bundles, externs, case files
  patch     patch sites: operand fields the host rewrites between runs, located by assembling twice, and the checked
            application through `ots-asm patch`
  pager     the instruction pager: a program whose queue images exceed `icu.iq_depth` becomes a fetch-driven one

The program generators and serving packages use these modules. Install them with `just python-setup`."""
