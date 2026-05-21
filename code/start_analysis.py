from Sports2D import Sports2D

if __name__ == "__main__":
    # 指向你刚才创建的配置文件
    config_path = r'E:\badminton\config.toml'
    
    print("--- 正在通过配置文件启动 Sports2D ---")
    # 最新版只接受配置文件路径作为参数
    Sports2D.process(config_path)