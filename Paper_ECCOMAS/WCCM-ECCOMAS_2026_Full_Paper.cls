%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ECCOMAS_2026.cls
%% LaTeX class for WCCM-ECCOMAS 2026 Full Paper
%% Reconstructed from the official Word full paper instructions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ECCOMAS_2026}[2026/05/11 WCCM-ECCOMAS 2026 full paper class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\LoadClass[12pt,a4paper]{article}

% ---------------------------------------------------------------------------
% Required packages
% ---------------------------------------------------------------------------
\RequirePackage{times}
\RequirePackage{geometry}
\RequirePackage{indentfirst}
\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb}
\RequirePackage{caption}
\RequirePackage{fancyhdr}
\RequirePackage{etoolbox}
\RequirePackage{cite}

% ---------------------------------------------------------------------------
% Page layout: 16 cm x 21 cm printing box, centered on A4
% Page numbers and running headers are outside/around the box.
% ---------------------------------------------------------------------------
\geometry{
  a4paper,
  textwidth=16cm,
  textheight=21cm,
  hcentering,
  top=2.2cm,
  headheight=30pt,
  headsep=12pt,
  footskip=32pt
}

\setlength{\parindent}{0.5cm}
\setlength{\parskip}{0pt}
\setlength{\baselineskip}{14pt}
\setlength{\textfloatsep}{12pt}
\setlength{\floatsep}{12pt}
\setlength{\intextsep}{12pt}
\setlength{\abovecaptionskip}{6pt}
\setlength{\belowcaptionskip}{12pt}
\setlength{\abovedisplayskip}{6pt}
\setlength{\belowdisplayskip}{6pt}
\setlength{\abovedisplayshortskip}{6pt}
\setlength{\belowdisplayshortskip}{6pt}

\renewcommand{\baselinestretch}{1.0}
\normalsize

% ---------------------------------------------------------------------------
% Metadata commands
% ---------------------------------------------------------------------------
\newcommand{\shortauthors}[1]{\gdef\@shortauthors{#1}}
\newcommand{\address}[1]{\gdef\@address{#1}}
\newcommand{\keywords}[1]{\gdef\@keywords{#1}}
\newcommand{\summary}[1]{\gdef\@summary{#1}}

\gdef\@shortauthors{}
\gdef\@address{}
\gdef\@keywords{}
\gdef\@summary{}

% ---------------------------------------------------------------------------
% Headers and footers
% ---------------------------------------------------------------------------
\fancypagestyle{ECCOMASfirst}{%
  \fancyhf{}
  \fancyhead[R]{%
    \fontsize{8}{9}\selectfont
    \begin{tabular}{r}
    17th World Congress on Computational Mechanics (WCCM)\\
    10th European Congress on Computational Methods in Applied Sciences and Engineering (ECCOMAS)\\
    19 - 24 July 2026, Munich, Germany
    \end{tabular}%
  }
  \renewcommand{\headrulewidth}{0pt}
  \fancyfoot[C]{}
}

\fancypagestyle{ECCOMAS}{%
  \fancyhf{}
  \fancyhead[C]{%
    \fontsize{10}{12}\selectfont \@shortauthors\\[-2pt]
    \rule{16cm}{0.4pt}%
  }
  \renewcommand{\headrulewidth}{0pt}
  \fancyfoot[C]{\thepage}
}

\pagestyle{ECCOMAS}

% ---------------------------------------------------------------------------
% Title block
% ---------------------------------------------------------------------------
\renewcommand{\maketitle}{%
  \thispagestyle{ECCOMASfirst}%
  \vspace*{0.0cm}%
  \begin{center}%
    {\fontsize{14}{16}\selectfont\bfseries\MakeUppercase{\@title}\par}%
    \vspace{5pt}%
    {\fontsize{12}{14}\selectfont\bfseries\MakeUppercase{\@author}\par}%
    \vspace{12pt}%
    {\fontsize{11}{13}\selectfont\@address\par}%
  \end{center}%
  \vspace{12pt}%
  \noindent{\bfseries Key words:} \@keywords\par%
  \vspace{12pt}%
  \noindent{\bfseries Summary.} \textit{\@summary}\par%
  \vspace{12pt}%
}

% ---------------------------------------------------------------------------
% Section and subsection formatting
% ---------------------------------------------------------------------------
\renewcommand\thesection{\arabic{section}}
\renewcommand\thesubsection{\thesection.\arabic{subsection}}

\renewcommand{\section}{%
  \@startsection{section}{1}{0pt}{12pt}{6pt}{\normalfont\fontsize{12}{14}\selectfont\bfseries\MakeUppercase}%
}

\renewcommand{\subsection}{%
  \@startsection{subsection}{2}{0pt}{12pt}{6pt}{\normalfont\fontsize{12}{14}\selectfont\bfseries}%
}

% ---------------------------------------------------------------------------
% Captions: centered, 10 pt, upper and lower case.
% ---------------------------------------------------------------------------
\captionsetup{
  font={small},
  labelfont=bf,
  labelsep=colon,
  justification=centering,
  singlelinecheck=true
}

% ---------------------------------------------------------------------------
% Tables
% ---------------------------------------------------------------------------
\renewcommand{\arraystretch}{1.05}

% ---------------------------------------------------------------------------
% References
% ---------------------------------------------------------------------------
\renewcommand{\refname}{REFERENCES}
\renewenvironment{thebibliography}[1]
{%
  \section*{REFERENCES}%
  \list{[\arabic{enumiv}]}%
       {\settowidth\labelwidth{[#1]}%
        \leftmargin\labelwidth
        \advance\leftmargin\labelsep
        \usecounter{enumiv}%
        \let\p@enumiv\@empty
        \renewcommand\theenumiv{\arabic{enumiv}}}%
  \sloppy
  \clubpenalty4000
  \widowpenalty4000
  \sfcode`\.\@m
}
{\endlist}

\endinput
