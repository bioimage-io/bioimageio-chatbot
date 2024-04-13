import sqlite3
import time

class QuotaManager:
    def __init__(self, db_file=':memory:', vip_list=None, default_quota=1.0, default_reset_period='daily'):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file)
        self.vip_list = vip_list or []
        self.default_quota = default_quota
        self.default_reset_period = self.period_to_seconds(default_reset_period)
        self._setup_database()

    def _setup_database(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quotas (
                user_id TEXT PRIMARY KEY, 
                quota REAL, 
                last_reset REAL,
                reset_period INTEGER,  -- Reset period in seconds
                max_quota REAL
            )
        ''')
        self.conn.commit()

    def set_user_quota(self, user_id, max_quota, reset_period):
        reset_seconds = self.period_to_seconds(reset_period)
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO quotas (user_id, quota, last_reset, reset_period, max_quota)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, max_quota, time.time(), reset_seconds, max_quota))
        self.conn.commit()

    def check_quota(self, user_id):
        if user_id in self.vip_list:
            return float('inf')

        cursor = self.conn.cursor()
        cursor.execute("SELECT quota, last_reset, reset_period, max_quota FROM quotas WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            quota, last_reset, reset_period, max_quota = row
            if time.time() - last_reset >= reset_period:
                quota = max_quota
                last_reset = time.time()
                cursor.execute("UPDATE quotas SET quota = ?, last_reset = ? WHERE user_id = ?",
                               (quota, last_reset, user_id))
                self.conn.commit()
        else:
            quota, last_reset, reset_period, max_quota = self.default_quota, time.time(), self.default_reset_period, self.default_quota
            cursor.execute("INSERT INTO quotas (user_id, quota, last_reset, reset_period, max_quota) VALUES (?, ?, ?, ?, ?)",
                           (user_id, quota, last_reset, reset_period, max_quota))
            self.conn.commit()
        return quota

    def use_quota(self, user_id, amount):
        if self.check_quota(user_id) >= amount:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE quotas SET quota = quota - ? WHERE user_id = ?",
                           (amount, user_id))
            self.conn.commit()
            return True
        else:
            return False

    def reset_quota(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE quotas SET quota = max_quota, last_reset = ? WHERE user_id = ?",
                       (time.time(), user_id))
        self.conn.commit()

    def period_to_seconds(self, period):
        if period == 'monthly':
            return 30 * 86400
        elif period == 'weekly':
            return 7 * 86400
        elif period == 'daily':
            return 86400
        elif period == 'hourly':
            return 3600
        else:
            raise ValueError("Invalid period. Choose from 'daily', 'weekly', 'monthly'.")

if __name__ == '__main__':
    # Testing the functionality with asserts
    quota_manager = QuotaManager(vip_list=['userVIP'], default_quota=2.0, default_reset_period='daily')
    # quota_manager.set_user_quota('user123', 1.0, 'daily')
    quota_manager.set_user_quota('user234', 10.0, 'weekly')
    quota_manager.set_user_quota('user345', 30.0, 'monthly')

    # Assert initial quotas are set correctly
    assert quota_manager.check_quota('userVIP') == float('inf'), "VIP user quota should be infinite"
    assert quota_manager.check_quota('user123') == 2.0, "user123 initial quota should be 2.0"
    assert quota_manager.check_quota('user234') == 10.0, "user234 initial quota should be 10.0"
    assert quota_manager.check_quota('user345') == 30.0, "user345 initial quota should be 30.0"

    # Assert usage and quota management
    assert quota_manager.use_quota('user123', 0.5) == True, "Should allow using 0.5 quota"
    assert quota_manager.check_quota('user123') == 2-0.5, "user123 quota after use should be 2-0.5"
    assert quota_manager.use_quota('user123', 3.6) == False, "Should not allow using 3.6 quota (not enough left)"
    assert quota_manager.use_quota('userVIP', 1000) == True, "VIP should always be allowed to use quota"
    assert quota_manager.check_quota('user234') == 10.0, "user234 should still have full quota"

    # Test manual reset
    quota_manager.reset_quota('user123')
    assert quota_manager.check_quota('user123') == 2.0, "user123 should have full quota after reset"

    # Display final test results
    print("Tests completed successfully.")
