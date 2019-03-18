import os
import subprocess

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.201.b09-2.el7_6.x86_64"
os.environ["SPARK_HOME"] = "/usr/hdp/3.1.0.0-78/spark2"
os.environ["HADOOP_HOME"] = "/usr/hdp/current/hadoop-client"
os.environ["HADOOP_USER_NAME"] = "hdfs"
os.environ["HADOOP_HDFS_HOME"] = "/usr/hdp/current/hadoop-hdfs-client"
os.environ["PATH"] = os.environ["PATH"] + os.pathsep + os.environ["SPARK_HOME"] + "/bin" + os.pathsep + os.environ["HADOOP_HOME"] + "/bin"  + os.pathsep + os.environ["HADOOP_HDFS_HOME"] + "/bin"
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6"
os.environ["SPARK_YARN_USER_ENV"] = "PYSPARK_PYTHON=/usr/bin/python3.6"
os.environ["LIB_HDFS"] = "/usr/hdp/3.1.0.0-78/usr/lib"
os.environ["LIB_JVM"] = "/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.201.b09-2.el7_6.x86_64/jre/lib/amd64/server"
os.environ["LD_LIBRARY_PATH"] = os.environ["LIB_HDFS"] + os.pathsep + os.environ["LIB_JVM"]
os.environ["CLASSPATH"] = subprocess.run(["hadoop", "classpath", "--glob"], stdout=subprocess.PIPE).stdout.decode("utf-8")

# if GPU enabled
# os.environ["LIB_CUDA"] = 
# os.environ["LD_LIBRARY_PATH"] = os.environ["LIB_HDFS"] + os.pathsep + os.environ["LIB_JVM"] + os.environ["LIB_CUDA"]