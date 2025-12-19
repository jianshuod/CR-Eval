FROM lscr.io/linuxserver/libreoffice:latest

RUN apk update && apk add parallel

COPY convert_doc.sh /convert_doc.sh
COPY run_parallel.sh /run_parallel.sh

RUN chmod +x /convert_doc.sh /run_parallel.sh

CMD ["/run_parallel.sh"]
