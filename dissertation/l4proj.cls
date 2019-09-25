% =============================================================================
% l4proj.cls
%
% Template for final year projects in the School of Computing Science at the
% University of Glasgow. Adapted from the level 3 project template originally
% provided by Colin Perkins <csp@dcs.gla.ac.uk>.
%
% =============================================================================

\ProvidesClass{l4proj}[%
2009/01/09 Colin Perkins <csp@dcs.gla.ac.uk>.%
2009/09/16 Modified by Tim Storer <tws@dcs.gla.ac.uk> to support non pdf modes.%
2011/02/15 Modified by Gregg Hamilton <hamiltgr@dcs.gla.ac.uk> for use in final year projects.%
2012/10/18 Modified by Patrick Prosser, use geometry to control margins, simplified.%
2018/09/14 Modified by John Williamson, improve typographic appearance%
]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}

\ProcessOptions
\LoadClass[10pt,a4paper]{report}

% Monospace font is Fira Mono Sans
% the body font used is FBB (Bembo)

\usepackage{iftex}

\ifPDFTeX
  \usepackage{fbb}
  \usepackage{FiraMono}
  \usepackage{FiraSans}
  \usepackage{textcomp}
  \usepackage[english]{babel}
  \usepackage[T1]{fontenc}
\else
  % make sure the fonts are actually set
  \usepackage{fontspec}
  \usepackage{polyglossia}
  \setromanfont[Path=./fonts/, BoldFont=fbb-Bold, UprightFont=fbb-Regular, ItalicFont=fbb-Italic, BoldItalicFont=fbb-BoldItalic]{fbb}
  \setmonofont[Path=./fonts/, BoldFont=FiraMono-Bold, UprightFont=FiraMono-Regular]{FiraMono.ttf}
  \setsansfont[Path=./fonts/, BoldFont=FiraSans-Bold, UprightFont=FiraSans-Regular, ItalicFont=FiraSans-Italic, BoldItalicFont=FiraSans-BoldItalic]{FiraSans.ttf}
\fi

\usepackage[libertine,bigdelims,vvarbb]{newtxmath} % bb from STIX
\usepackage[cal=boondoxo]{mathalfa} % mathcal

\usepackage{microtype} % improve typography
\usepackage{anyfontsize} % allow any size of font
\usepackage{amsmath,amsfonts,amsbsy,amssymb}
\usepackage[style]{abstract} % abstract styling

% set bibliography styling
\usepackage{natbib}
\bibpunct{(}{)}{;}{a}{}{;}


\usepackage{etoolbox}
\usepackage[compact]{titlesec} % section heading formatting
\usepackage{color} % colors
\usepackage[table,svgnames]{xcolor}
\usepackage{parskip} % adjustable line spacing
\usepackage{listings} % nice code listings

% remove dots from ToC
\usepackage[titles]{tocloft} 
\renewcommand{\cftdot}{}
\usepackage[nottoc]{tocbibind}

% from: https://gist.github.com/FelipeCortez/10729134
% set up listings for prettier output
\definecolor{bluekeywords}{rgb}{0.13, 0.19, 0.7}
\definecolor{greencomments}{rgb}{0.1, 0.5, 0.2}
\definecolor{redstrings}{rgb}{0.8, 0.15, 0.1}
\definecolor{graynumbers}{rgb}{0.5, 0.5, 0.5}
\definecolor{subtlegray}{rgb}{0.98, 0.98, 0.98}
\usepackage{lstautogobble}
\usepackage{listings}
\lstset{
    autogobble,    
    columns=fullflexible,
    showspaces=false,
    showtabs=false,
    breaklines=true,
    showstringspaces=false,
    breakatwhitespace=true,
    escapeinside={(*@}{@*)},
    rulecolor=\color{lightgray},
    backgroundcolor=\color{subtlegray},
    commentstyle=\color{greencomments},
    keywordstyle=\color{bluekeywords},
    stringstyle=\color{redstrings},
    numberstyle=\color{graynumbers},
    basicstyle=\ttfamily\linespread{1.15}\footnotesize,
    frame=tb,
    framesep=12pt,
    framexleftmargin=12pt,
    tabsize=4,
    captionpos=b
}
%% ---

% nice number printing
\usepackage{numprint}

% allow PDF graphics
\usepackage{graphicx}

% allow section styling
%\usepackage{sectsty}

% make all cross-references clickable
\usepackage[hidelinks]{hyperref}

% enable styled captions
\usepackage[format=plain,
labelfont={bf,it},
textfont=it,tableposition=above]{caption}

\newcommand{\secfont}{\normalfont\sffamily}
% % style the sections and the abstract
% \allsectionsfont{\secfont}


\renewcommand{\abstractnamefont}{\Large\secfont}
\renewcommand{\abstitlestyle}[1]{{{\let\clearpage\relax \chapter*{#1}}}}

\usepackage[toc]{appendix}

% allow subfloats
\usepackage{subcaption}


% fix the margins and linespacing
\usepackage[top=3.5cm,bottom=4.0cm,left=3.5cm,right=3.5cm]{geometry}
\setlength{\parskip}{5pt plus 2pt}

% define the title page
\renewcommand\maketitle{
  \begin{titlepage}%
      \definecolor{UniBlue}{HTML}{00355F}
        
      \let\footnotesize\small
      \let\footnoterule\relax
      \let\footnote \thanks

      % the logo, full page width
      \vspace{1in}
      \includegraphics[width=\linewidth]{images/CompSci_colour.pdf}
      \vskip 0.5em
      \color{UniBlue}
      {\large \sffamily Honours Individual Project Dissertation}\\ 

      % title and author
      
      \vspace{2cm}
        \Large
        \vskip 8em%
      
        \begin{center}
          
          \color{UniBlue}
            \noindent
            {\Huge \scshape  \@title \par}%
            
            
            
            
          
        \end{center}
        \vfill
        {\Large
        \textbf{\@author}\\
        \@date}
            

    
  \end{titlepage}%

  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@consentname\@empty
  \global\let\@consentdate\@empty

  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\usepackage{booktabs}
% set up headers and footers
% no footer, page numbers at top-right
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhead{}
\fancyfoot{}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\fancypagestyle{plain}{%
    \renewcommand{\headrulewidth}{0pt}%
    \fancyhf{}%
    \fancyhead[R]{\thepage}%
}

\usepackage{titlesec}
% set up chapter headings
\usepackage{xstring}
\titleformat{\chapter}[display]
  % only show numbers if they aren't Chapter 0 (i.e. front matter)
  {\normalfont}{}{0pt}{
    \huge\bfseries\secfont
    \IfInteger{\thechapter}{\ifnum0=0\thechapter\relax\else\relax {\Huge \thechapter}\ \rule[-0.25em]{0.5pt}{1.5em}\ \fi}
    {\thechapter \ \rule[-0.25em]{0.5pt}{1.5em}\ }}
  %[\titlerule]
  \titlespacing*{\chapter}      
  {0pt}{0pt}{15pt}  

% Educational consent form
\newcommand{\educationalconsent}{
    \ifdefined\consentname
          \newpage
          \chapter*{Education Use Consent}

          I hereby grant my permission for this project to be stored, distributed and shown to other
          University of Glasgow students and staff for educational purposes. 
          \textbf{Please note that you are under no obligation to sign 
          this declaration, but doing so would help future students.}

          \begin{tabular}{@{}llll}
                &                     &            &                     \\
                &                     &            &                     \\
          Signature: & \consentname & Date: & \consentdate \\
                &                     &            &                     \\
                &                     &            &                     \\
          \end{tabular}
    \else
    \newpage
        \chapter*{Education Use Consent}
        Consent for educational reuse withheld. Do not distribute.
    \fi 

}

% highlight boxes
\usepackage{tcolorbox}
\newenvironment{highlight}
{\begin{tcolorbox}[notitle,boxrule=0pt,colback=green!10,colframe=green!10]}
{
\end{tcolorbox}
}

\newenvironment{highlight_title}[1]
{\begin{tcolorbox}[title=#1,boxrule=2pt,colback=green!10,colframe=green!20,coltitle=black,fonttitle=\bfseries]}
{
\end{tcolorbox}
}



% adjust margins
\setlength{\parindent}{0pt}


% use (a) in figures
\captionsetup{subrefformat=parens}

% Algorithm typesetting package
\usepackage[plain, noline, shortend, linesnumberedhidden]{algorithm2e}
\SetAlFnt{\sffamily \small}

% adjust spacing
\titlespacing{\section}{10pt}{1.5\baselineskip}{8pt plus 3pt minus 3pt}
\titlespacing{\subsection}{10pt}{\baselineskip}{8pt plus 4pt minus 1pt}
\titleformat{\subsubsection}[runin]{\bfseries }{}{}{}[]
\titlespacing{\chapter}{0pt}{0pt}{40pt plus 10pt minus 5pt} 

% switch to roman numeral for frontmatter
\pagenumbering{roman}

% fix itemise so it looks reasonable
\renewcommand{\@listI}{%
      \leftmargin=25pt
      \rightmargin=0pt
      \labelsep=5pt
      \labelwidth=20pt
      \itemindent=0pt
      \listparindent=0pt
      \topsep=0pt plus 2pt minus 4pt
      \partopsep=0pt plus 1pt minus 1pt
      \parsep=1pt plus 1pt
      \itemsep=\parsep}

