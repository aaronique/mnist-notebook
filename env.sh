export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.201.b09-2.el7_6.x86_64 \
export SPARK_HOME=/usr/hdp/3.1.0.0-78/spark2 \
export HADOOP_HOME=/usr/hdp/current/hadoop-client \
export HADOOP_USER_NAME=hdfs \
export HADOOP_HDFS_HOME=/usr/hdp/current/hadoop-hdfs-client \
export PATH=${PATH}:${SPARK_HOME}/bin:${HADOOP_HOME}/bin:${HADOOP_HDFS_HOME}/bin \
export PYSPARK_PYTHON=/usr/bin/python3.6 \
export SPARK_YARN_USER_ENV="PYSPARK_PYTHON=/usr/bin/python3.6" \
export LIB_HDFS=/usr/hdp/3.1.0.0-78/usr/lib \
export LIB_JVM=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.201.b09-2.el7_6.x86_64/jre/lib/amd64/server \
export LD_LIBRARY_PATH=${LIB_HDFS}:${LIB_JVM} \
export CLASSPATH=$(hadoop classpath --glob)

# if GPU enabled
# export LIB_CUDA =  \
# export LD_LIBRARY_PATH = ${LIB_HDFS}:${LIB_JVM}:${LIB_CUDA}
